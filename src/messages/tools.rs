use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::message::{FunctionCall, FunctionResponse};

#[derive(Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    description: String,
    parameters: serde_json::Value,
    #[serde(skip)]
    pub handler: Option<Arc<dyn ErasedToolHandler>>,
}

impl fmt::Debug for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .finish_non_exhaustive()
    }
}

impl Tool {
    // Make new tool. Tool needs to know what kind of stuff it works with.
    #[must_use]
    pub fn builder() -> ToolBuilder {
        ToolBuilder::default()
    }

    // Use tool on something.
    #[must_use]
    pub fn invoke(&self, input: serde_json::Value, cx: ToolContext) -> Option<FunctionResponse> {
        // If tool has a way to do things...
        if let Some(handler) = &self.handler {
            // ...try to do the thing.
            match handler.call(input, cx) {
                // If thing worked, return good result.
                Ok(response) => Some(FunctionResponse {
                    name: self.name.clone(),
                    response,
                }),
                // If thing failed, return error result.
                Err(err) => Some(FunctionResponse {
                    name: self.name.clone(),
                    response: serde_json::to_value(err).unwrap(),
                }),
            }
        // If tool has no way to do things, return nothing.
        } else {
            None
        }
    }
}

#[derive(Default)]
pub struct ToolBuilder {
    name: Option<String>,
    description: Option<String>,
    parameters: Option<serde_json::Value>,
    handler: Option<Arc<dyn ErasedToolHandler>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolBuilderError {
    #[error("Tool name is required")]
    MissingName,
    #[error("Tool description is required")]
    MissingDescription,
    #[error("Tool parameters is required")]
    MissingParameters,
}

impl ToolBuilder {
    #[must_use]
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    #[must_use]
    pub fn description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    #[must_use]
    pub fn handler<H, T, R>(mut self, handler: H) -> Self
    where
        H: ToolHandler<T, R> + Send + Sync + 'static,
        T: JsonSchema + DeserializeOwned + Send + Sync + 'static,
        R: Serialize + Send + Sync + 'static,
    {
        let settings = schemars::gen::SchemaSettings::openapi3().with(|s| {
            s.inline_subschemas = true;
            s.meta_schema = None;
        });
        let gen = schemars::gen::SchemaGenerator::new(settings);
        let json_schema = gen.into_root_schema_for::<T>();
        let mut parameters = serde_json::to_value(json_schema).unwrap();
        parameters.as_object_mut().unwrap().remove("title").unwrap();
        self.parameters = Some(parameters);

        let wrapper = ToolHandlerWrapper::<H, T, R> {
            handler,
            phantom: PhantomData,
        };
        self.handler = Some(Arc::new(wrapper));
        self
    }

    #[must_use]
    pub fn props<T>(mut self) -> Self
    where
        T: JsonSchema + DeserializeOwned + Send + Sync + 'static,
    {
        let settings = schemars::gen::SchemaSettings::openapi3().with(|s| {
            s.inline_subschemas = true;
            s.meta_schema = None;
        });
        let gen = schemars::gen::SchemaGenerator::new(settings);
        let json_schema = gen.into_root_schema_for::<T>();
        let mut parameters = serde_json::to_value(json_schema).unwrap();
        parameters.as_object_mut().unwrap().remove("title").unwrap();
        self.parameters = Some(parameters);

        self
    }

    /// Consumes the builder, returning a [`Tool`] if all required fields have been set.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` or `description` are not set.
    pub fn build(self) -> Result<Tool, ToolBuilderError> {
        let name = self.name.ok_or(ToolBuilderError::MissingName)?;
        let description = self
            .description
            .ok_or(ToolBuilderError::MissingDescription)?;
        let parameters = self.parameters.ok_or(ToolBuilderError::MissingParameters)?;

        Ok(Tool {
            name,
            description,
            parameters,
            handler: self.handler,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    pub resources: Arc<Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>>,
}

impl ToolContext {
    pub fn push_resource<T: Any + Send + Sync + Clone>(&mut self, resource: T) {
        self.resources
            .lock()
            .unwrap()
            .insert(TypeId::of::<T>(), Arc::new(Mutex::new(resource)));
    }

    #[must_use]
    pub fn add_resource<T: Any + Send + Sync + Clone>(mut self, resource: T) -> Self {
        self.push_resource(resource);
        self
    }

    #[must_use]
    pub fn get_resource<T: Any + Send + Sync + Clone>(&self) -> Option<T> {
        self.resources
            .lock()
            .unwrap()
            .get(&TypeId::of::<T>())
            .and_then(|boxed_resource| boxed_resource.downcast_ref::<T>().cloned())
    }

    #[must_use]
    pub fn expect_resource<T: Any + Send + Sync + Clone>(&self) -> T {
        self.get_resource().expect("Resource not found")
    }
}

pub trait FromContext: Sized + Send + 'static {
    fn from_context(context: &ToolContext) -> Option<Self>;
}

impl<T> FromContext for T
where
    T: Any + Send + Sync + Clone,
{
    fn from_context(context: &ToolContext) -> Option<Self> {
        context.get_resource::<T>()
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Tools {
    tools: HashMap<String, Tool>,
    #[serde(skip)]
    context: ToolContext,
}

impl Serialize for Tools {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        #[derive(Serialize)]
        #[serde(rename = "camelCase")]
        struct SerializedTools<'a> {
            function_declarations: Vec<&'a Tool>,
        }
        let tools = SerializedTools {
            function_declarations: self.tools.values().collect(),
        };
        tools.serialize(serializer)
    }
}

impl Tools {
    pub fn add<T: Into<Tool>>(&mut self, tool: T) {
        let tool = tool.into();
        self.tools.insert(tool.name.clone(), tool);
    }

    #[must_use]
    pub fn with_tool<T: Into<Tool>>(mut self, tool: T) -> Self {
        self.add(tool);
        self
    }

    #[must_use]
    pub fn get_tool(&self, tool_name: &str) -> Option<&Tool> {
        self.tools.get(tool_name)
    }

    #[must_use]
    pub fn invoke(&self, function_call: &FunctionCall) -> Option<FunctionResponse> {
        if let Some(tool) = self.get_tool(&function_call.name) {
            tool.invoke(
                function_call
                    .args
                    .as_ref()
                    .unwrap_or(&serde_json::Value::Null)
                    .clone(),
                self.context.clone(),
            )
        } else {
            None
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn push_resource<T: Any + Send + Sync + Clone>(&mut self, resource: T) {
        self.context.push_resource(resource);
    }

    #[must_use]
    pub fn with_resource<T: Any + Send + Sync + Clone>(mut self, resource: T) -> Self {
        self.context.push_resource(resource);
        self
    }
}

impl From<Tool> for Tools {
    fn from(tool: Tool) -> Self {
        Tools::default().with_tool(tool)
    }
}

impl From<Vec<Tool>> for Tools {
    fn from(tools: Vec<Tool>) -> Self {
        let mut tools_map = Tools::default();
        for tool in tools {
            tools_map.add(tool);
        }
        tools_map
    }
}

#[derive(Debug, thiserror::Error, Serialize)]
#[error("{error}")]
pub struct HandlerError {
    error: String,
}

impl HandlerError {
    pub fn new<T: ToString>(e: T) -> Self {
        Self {
            error: e.to_string(),
        }
    }
}

pub trait ToolHandler<T, R> {
    fn call(&self, input: T, cx: &ToolContext) -> Result<R, HandlerError>;
}

impl<T, R, F> ToolHandler<T, R> for F
where
    F: Fn(T, &ToolContext) -> Result<R, HandlerError>,
    R: Serialize,
{
    fn call(&self, input: T, cx: &ToolContext) -> Result<R, HandlerError> {
        (self)(input, cx)
    }
}

pub trait ErasedToolHandler: Send + Sync {
    fn call(
        &self,
        input: serde_json::Value,
        cx: ToolContext,
    ) -> Result<serde_json::Value, HandlerError>;
}

#[derive(Debug, Clone)]
pub struct ToolHandlerWrapper<H, T, R>
where
    H: ToolHandler<T, R>,
{
    pub handler: H,
    pub phantom: PhantomData<(T, R)>,
}

impl<H, T, R> ErasedToolHandler for ToolHandlerWrapper<H, T, R>
where
    H: ToolHandler<T, R> + Send + Sync,
    T: JsonSchema + DeserializeOwned + Send + Sync,
    R: Serialize + Send + Sync,
{
    fn call(
        &self,
        input: serde_json::Value,
        cx: ToolContext,
    ) -> Result<serde_json::Value, HandlerError> {
        let props: T = serde_json::from_value(input).map_err(|e| HandlerError {
            error: e.to_string(),
        })?;
        let result = self.handler.call(props, &cx)?;
        serde_json::to_value(result).map_err(|e| HandlerError {
            error: e.to_string(),
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

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};

    use super::*;

    #[derive(Serialize, Deserialize, JsonSchema)]
    struct TestParams {
        foo: String,
        bar: i32,
    }

    #[derive(Serialize)]
    struct TestResponse {
        message: String,
    }

    fn test_tool_handler(
        params: TestParams,
        _cx: &ToolContext,
    ) -> Result<TestResponse, HandlerError> {
        Ok(TestResponse {
            message: format!("foo: {}, bar: {}", params.foo, params.bar),
        })
    }

    #[test]
    fn test_tool_builder() {
        let tool = Tool::builder()
            .name("test_tool".to_string())
            .description("This is a test tool.".to_string())
            .handler(|params: TestParams, _cx: &ToolContext| {
                Ok(TestResponse {
                    message: format!("foo: {}, bar: {}", params.foo, params.bar),
                })
            })
            .build()
            .unwrap();

        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.description, "This is a test tool.");

        let parameters = tool.parameters;
        assert_eq!(parameters["type"], Value::String("object".to_string()));
        assert_eq!(
            parameters["properties"]["foo"]["type"],
            Value::String("string".to_string())
        );
        assert_eq!(
            parameters["properties"]["bar"]["type"],
            Value::String("integer".to_string())
        );
    }

    #[test]
    fn test_tool_serialization() {
        let tool = Tool {
            name: "test_tool".to_string(),
            description: "This is a test tool.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "foo": {
                        "type": "string"
                    },
                    "bar": {
                        "type": "integer"
                    }
                }
            }),
            handler: None,
        };

        let serialized = serde_json::to_string(&tool).unwrap();
        let expected = json!({
            "name": "test_tool",
            "description": "This is a test tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "foo": {
                        "type": "string"
                    },
                    "bar": {
                        "type": "integer"
                    }
                }
            }
        });
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&serialized).unwrap(),
            expected
        );
    }

    #[test]
    fn test_tool_deserialization() {
        let json_str = r#"
        {
            "name": "test_tool",
            "description": "This is a test tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "foo": {
                        "type": "string"
                    },
                    "bar": {
                        "type": "integer"
                    }
                }
            }
        }
        "#;
        let expected_tool = Tool {
            name: "test_tool".to_string(),
            description: "This is a test tool.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "foo": {
                        "type": "string"
                    },
                    "bar": {
                        "type": "integer"
                    }
                }
            }),
            handler: None,
        };
        let deserialized_tool: Tool = serde_json::from_str(json_str).unwrap();
        assert_eq!(deserialized_tool.name, expected_tool.name);
        assert_eq!(deserialized_tool.description, expected_tool.description);
        assert_eq!(deserialized_tool.parameters, expected_tool.parameters);
    }

    #[test]
    fn test_tool_call() {
        let tool = Tool::builder()
            .name("test_tool".to_string())
            .description("This is a test tool.".to_string())
            .handler(test_tool_handler)
            .build()
            .unwrap();

        let input = json!({
            "foo": "hello",
            "bar": 42,
        });

        let cx = ToolContext::default();

        let response = tool.invoke(input, cx).unwrap();

        assert_eq!(response.name, "test_tool");
        assert_eq!(
            response.response,
            json!({
                "message": "foo: hello, bar: 42"
            })
        );
    }

    #[test]
    fn test_tool_call_error() {
        struct ErrorToolHandler;

        impl ToolHandler<TestParams, TestResponse> for ErrorToolHandler {
            fn call(
                &self,
                _params: TestParams,
                _cx: &ToolContext,
            ) -> Result<TestResponse, HandlerError> {
                Err(HandlerError {
                    error: "Something went wrong!".to_string(),
                })
            }
        }

        let tool = Tool::builder()
            .name("error_tool".to_string())
            .description("This tool always returns an error.")
            .handler(ErrorToolHandler)
            .build()
            .unwrap();

        let input = json!({
            "foo": "hello",
            "bar": 42,
        });

        let cx = ToolContext::default();

        let response = tool.invoke(input, cx).unwrap();

        assert_eq!(response.name, "error_tool");
        assert_eq!(
            response.response,
            json!({
                "error": "Something went wrong!"
            })
        );
    }
}
