use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt,
    marker::PhantomData,
    sync::{Arc, Mutex, RwLock},
};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};

use super::message::{FunctionCall, FunctionResponse};

#[async_trait]
pub trait AnyTool: Send + Sync {
    fn name(&self) -> String;
    fn description(&self) -> Option<String>;
    async fn invoke_any(&self, function_call: FunctionCall) -> FunctionResponse;
    fn input_schema(&self) -> Value;
}

#[async_trait]
pub trait Tool: Clone + Send + Sync {
    type Input: JsonSchema + DeserializeOwned + Send + Sync;
    type Output: Serialize + Send + Sync;
    type Error: ToString;
    fn name(&self) -> String;
    fn description(&self) -> Option<String> {
        None
    }
    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    fn input_schema(&self) -> Value {
        let settings = schemars::gen::SchemaSettings::openapi3().with(|s| {
            s.inline_subschemas = true;
            s.meta_schema = None;
        });
        let gen = schemars::gen::SchemaGenerator::new(settings);
        let json_schema = gen.into_root_schema_for::<Self::Input>();
        let mut input_schema = serde_json::to_value(json_schema).unwrap();
        input_schema
            .as_object_mut()
            .unwrap()
            .remove("title")
            .unwrap();
        if input_schema.get("properties").is_some() {
            input_schema
        } else {
            serde_json::json!(None::<()>)
        }
    }
}

#[async_trait]
impl<T: Tool + Send + Sync> AnyTool for T {
    fn name(&self) -> String {
        self.name()
    }

    fn description(&self) -> Option<String> {
        self.description()
    }

    async fn invoke_any(&self, function_call: FunctionCall) -> FunctionResponse {
        if let Some(input) = function_call.args {
            let typed_input: T::Input = match serde_json::from_value(input) {
                Ok(input) => input,
                Err(e) => {
                    return FunctionResponse {
                        name: function_call.name,
                        response: FunctionCallError::InputDeserializationFailed(e.to_string())
                            .to_string()
                            .into(),
                    }
                }
            };

            match self.invoke(typed_input).await {
                Ok(output) => match serde_json::to_value(output) {
                    Ok(value) => FunctionResponse {
                        name: function_call.name,
                        response: value,
                    },
                    Err(e) => FunctionResponse {
                        name: function_call.name,
                        response: FunctionCallError::OutputSerializationFailed(e.to_string())
                            .to_string()
                            .into(),
                    },
                },
                Err(e) => FunctionResponse {
                    name: function_call.name,
                    response: e.to_string().into(),
                },
            }
        } else {
            unimplemented!()
        }
    }

    fn input_schema(&self) -> Value {
        self.input_schema()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetadataInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Clone, Default)]
pub struct ToolBox {
    tools: Arc<RwLock<std::collections::HashMap<String, Arc<dyn AnyTool>>>>,
}

impl fmt::Debug for ToolBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tools = self.tools.read().map_err(|_| fmt::Error)?;
        f.debug_struct("ToolBox")
            .field("tools", &format!("HashMap with {} entries", tools.len()))
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FunctionCallError {
    #[error("Failed to execute tool: {0}")]
    ExecutionFailed(String),
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Failed to deserialize input: {0}")]
    InputDeserializationFailed(String),
    #[error("Failed to serialize output: {0}")]
    OutputSerializationFailed(String),
    #[error("Failed to generate input schema: {0}")]
    SchemaGenerationFailed(String),
}

#[derive(Serialize, Deserialize)]
pub struct FunctionDeclarations {
    function_declarations: Vec<ToolMetadataInfo>,
}

impl ToolBox {
    pub fn add<T: Tool + 'static>(&self, tool: T) {
        let name = tool.name().to_string();
        self.tools.write().unwrap().insert(name, Arc::new(tool));
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn AnyTool>> {
        self.tools.read().unwrap().get(name).cloned()
    }

    pub async fn invoke(&self, function_call: FunctionCall) -> FunctionResponse {
        match self.get(&function_call.name) {
            Some(tool) => tool.invoke_any(function_call).await,
            None => FunctionResponse {
                name: function_call.name.clone(),
                response: FunctionCallError::ToolNotFound(function_call.name)
                    .to_string()
                    .into(),
            },
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.read().unwrap().is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.read().unwrap().len()
    }

    #[must_use]
    pub fn metadata(&self) -> Vec<FunctionDeclarations> {
        let tools = self
            .tools
            .read()
            .unwrap()
            .values()
            .map(|tool| ToolMetadataInfo {
                name: tool.name(),
                description: tool.description(),
                parameters: tool.input_schema(),
            })
            .collect();

        vec![FunctionDeclarations {
            function_declarations: tools,
        }]
    }
}

impl Serialize for ToolBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        self.metadata().serialize(serializer)
    }
}

impl<T: Tool + 'static> From<Vec<T>> for ToolBox {
    fn from(tools: Vec<T>) -> Self {
        let toolbox = ToolBox::default();
        for tool in tools {
            toolbox.add(tool);
        }
        toolbox
    }
}

impl FromIterator<Arc<dyn AnyTool>> for ToolBox {
    fn from_iter<I: IntoIterator<Item = Arc<dyn AnyTool>>>(iter: I) -> Self {
        let toolbox = ToolBox::default();
        for tool in iter {
            toolbox
                .tools
                .write()
                .unwrap()
                .insert(tool.name().to_string(), tool);
        }
        toolbox
    }
}

#[derive(Debug, Clone)]
pub struct FunctionCallBuilder {
    name: String,
    args: String,
}

impl FunctionCallBuilder {
    pub fn new<T: Into<String>>(name: T) -> Self {
        Self {
            name: name.into(),
            args: String::new(),
        }
    }

    pub fn push_str(&mut self, s: &str) -> &mut Self {
        self.args.push_str(s);
        self
    }

    pub fn build(self) -> Result<FunctionCall, serde_json::Error> {
        Ok(FunctionCall {
            name: self.name,
            args: if self.args.trim().is_empty() {
                None
            } else {
                Some(serde_json::from_str(&self.args)?)
            },
        })
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename = "camelCase")]
pub struct ToolConfig {
    /// Function calling config.
    function_calling_config: Option<FunctionCallingConfig>,
}

impl ToolConfig {
    /// Set the function calling config.
    #[must_use]
    pub fn function_calling_config(
        mut self,
        function_calling_config: FunctionCallingConfig,
    ) -> Self {
        self.function_calling_config = Some(function_calling_config);
        self
    }

    #[must_use]
    pub fn mode(self, mode: Mode) -> Self {
        let fcc = FunctionCallingConfig::default().mode(mode);
        self.function_calling_config(fcc)
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename = "camelCase")]
pub struct FunctionCallingConfig {
    /// Specifies the mode in which function calling should execute.
    mode: Option<Mode>,
    /// A set of function names that, when provided, limits the functions the model will call.
    allowed_function_names: Option<Vec<String>>,
}

impl FunctionCallingConfig {
    /// Create a new FunctionCallingConfig with default values.
    pub fn new() -> Self {
        Self {
            mode: None,
            allowed_function_names: None,
        }
    }

    /// Set the function calling mode.
    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Set the allowed function names.
    pub fn allowed_function_names(mut self, allowed_function_names: Vec<String>) -> Self {
        self.allowed_function_names = Some(allowed_function_names);
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename = "camelCase")]
pub enum Mode {
    /// Unspecified function calling mode. This value should not be used.
    ModeUnspecified,
    /// Default model behavior, model decides to predict either a function call or a natural
    /// language response.
    Auto,
    /// Model is constrained to always predicting a function call only. If
    /// "allowedFunctionNames" are set, the predicted function call will be limited to any
    /// one of "allowedFunctionNames", else the predicted function call will be any one of
    /// the provided "functionDeclarations".
    Any,
    /// Model will not predict any function call. Model behavior is same as when not passing
    /// any function declarations.
    None,
}
