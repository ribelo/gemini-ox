use message::{Content, Contents, FunctionCall, Part, Parts};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::{Stream, StreamExt};
use tools::{Tool, ToolConfig, Tools};

use crate::{Gemini, GenerationConfig, SafetyRating, SafetySettings, BASE_URL};

pub mod message;
pub mod tools;

#[derive(Debug, Serialize)]
pub struct GenerateContentRequest {
    contents: Contents,
    #[serde(skip_serializing_if = "Tools::is_empty")]
    tools: Tools,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_settings: Option<SafetySettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    model: String,
    #[serde(skip)]
    gemini: Gemini,
}

/// Possible errors during the construction of `GenerateContentRequest`.
#[derive(Debug, thiserror::Error)]
pub enum GenerateContentRequestBuilderError {
    /// Indicates that no content was provided for the request.
    #[error("Content is required for GenerateContentRequest")]
    MissingContent,
    #[error("Model is required for GenerateContentRequest")]
    MissingModel,
}

#[derive(Debug)]
pub struct GenerateContentRequestBuilder {
    contents: Option<Contents>,
    tools: Option<Tools>,
    tool_config: Option<ToolConfig>,
    safety_settings: SafetySettings,
    system_instruction: Option<Content>,
    generation_config: Option<GenerationConfig>,
    model: Option<String>,
    gemini: Gemini,
}

impl GenerateContentRequestBuilder {
    pub fn new(gemini: Gemini) -> Self {
        GenerateContentRequestBuilder {
            contents: None,
            tools: None,
            tool_config: None,
            safety_settings: SafetySettings::default(),
            system_instruction: None,
            generation_config: None,
            model: None,
            gemini,
        }
    }
    #[must_use]
    pub fn contents<T: Into<Contents>>(mut self, contents: T) -> Self {
        self.contents = Some(contents.into());
        self
    }

    #[must_use]
    pub fn add_content<T: Into<Content>>(mut self, content: T) -> Self {
        let mut messages = self.contents.take().unwrap_or_default();
        messages.push(content);
        self.contents = Some(messages);
        self
    }

    #[must_use]
    pub fn model<T: Into<String>>(mut self, model: T) -> Self {
        self.model = Some(model.into());
        self
    }

    #[must_use]
    pub fn tools<T: Into<Tools>>(mut self, tools: T) -> Self {
        self.tools = Some(tools.into());
        self
    }

    #[must_use]
    pub fn tool_config<T: Into<ToolConfig>>(mut self, cfg: T) -> Self {
        self.tool_config = Some(cfg.into());
        self
    }

    #[must_use]
    pub fn add_tool<T: Into<Tool>>(mut self, tool: T) -> Self {
        let mut tools = self.tools.take().unwrap_or_default();
        tools.add(tool);
        self.tools = Some(tools);
        self
    }

    #[must_use]
    pub fn safety_settings(mut self, safety_settings: SafetySettings) -> Self {
        self.safety_settings = safety_settings;
        self
    }

    /// Sets the system instruction for the request.
    ///
    /// # Arguments
    ///
    /// * `system_instruction` - The system instruction for the request.
    #[must_use]
    pub fn system_instruction<T: Into<Content>>(mut self, system_instruction: T) -> Self {
        self.system_instruction = Some(system_instruction.into());
        self
    }

    /// Sets the generation configuration for the request.
    ///
    /// # Arguments
    ///
    /// * `generation_config` - The generation configuration for the request.
    #[must_use]
    pub fn generation_config(mut self, generation_config: GenerationConfig) -> Self {
        self.generation_config = Some(generation_config);
        self
    }

    pub fn build(self) -> Result<GenerateContentRequest, GenerateContentRequestBuilderError> {
        let contents = self
            .contents
            .ok_or(GenerateContentRequestBuilderError::MissingContent)?;
        if contents.is_empty() {
            return Err(GenerateContentRequestBuilderError::MissingContent);
        }
        let model = self
            .model
            .ok_or(GenerateContentRequestBuilderError::MissingModel)?;
        Ok(GenerateContentRequest {
            contents,
            tools: self.tools.unwrap_or_default(),
            safety_settings: Some(self.safety_settings),
            system_instruction: self.system_instruction,
            generation_config: self.generation_config,
            model,
            gemini: self.gemini,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GenerateContentRequestError {
    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),
    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),
    #[error("Invalid request error: {message}")]
    InvalidRequestError {
        code: Option<String>,
        details: serde_json::Value,
        message: String,
        status: Option<String>,
    },
    #[error(transparent)]
    EventSourceError(#[from] reqwest_eventsource::Error),
}

impl Gemini {
    #[must_use]
    pub fn generate_content(&self) -> GenerateContentRequestBuilder {
        GenerateContentRequestBuilder::new(self.clone())
    }
}

impl GenerateContentRequest {
    pub async fn send(&self) -> Result<GenerateContentResponse, GenerateContentRequestError> {
        // Create the request URL.
        let url = format!(
            "{}/{}/models/{}:generateContent?key={}",
            BASE_URL, self.gemini.api_version, self.model, self.gemini.api_key
        );
        // Send the request.
        let res = self.gemini.client.post(&url).json(self).send().await?;

        // Check the status code.
        match res.status().as_u16() {
            200 | 201 => {
                let data: GenerateContentResponse = res.json().await?;
                Ok(data)
            }
            _status => {
                let mut e: Value = res.json().await?;
                Err(GenerateContentRequestError::InvalidRequestError {
                    code: e["error"]["code"].as_str().map(String::from),
                    details: e["error"]["details"].take(),
                    message: e["error"]["message"]
                        .as_str()
                        .map_or_else(|| "no message".to_string(), String::from),
                    status: e["error"]["status"].as_str().map(String::from),
                })
            }
        }
    }
    pub fn stream(
        &self,
    ) -> impl Stream<Item = Result<GenerateContentResponse, GenerateContentRequestError>> {
        // Create the request URL.
        let url = format!(
            "{}/{}/models/{}:streamGenerateContent?alt=sse&key={}",
            BASE_URL, self.gemini.api_version, self.model, self.gemini.api_key
        );
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut es = self
            .gemini
            .client
            .post(&url)
            .json(self)
            .eventsource()
            .unwrap();

        tokio::spawn(async move {
            while let Some(event) = es.next().await {
                match event {
                    Ok(Event::Open) => {}
                    Ok(Event::Message(msg)) => {
                        let res = serde_json::from_str::<GenerateContentResponse>(&msg.data)
                            .map_err(GenerateContentRequestError::SerdeError);
                        tx.send(res).unwrap();
                    }
                    Err(err) => {
                        if matches!(err, reqwest_eventsource::Error::StreamEnded) {
                            es.close();
                            break;
                        }
                        tx.send(Err(GenerateContentRequestError::EventSourceError(err)))
                            .unwrap();
                    }
                }
            }
        });

        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    pub candidates: Vec<ResponseCandidate>,
    pub prompt_feedback: Option<PromptFeedback>,
    pub usage_metadata: Option<UsageMetadata>,
}

impl GenerateContentResponse {
    #[must_use]
    pub fn content(&self) -> Option<&Content> {
        self.candidates.first().map(|c| &c.content)
    }
    #[must_use]
    pub fn get_function_calls(&self) -> Vec<&FunctionCall> {
        self.content()
            .map(|c| c.parts().get_function_calls())
            .unwrap_or_default()
    }
    #[must_use]
    pub fn invoke_functions(&self, tools: &Tools) -> Content {
        let mut content = Content::user();
        for fc in self.get_function_calls() {
            if let Some(res) = tools.invoke(fc) {
                content.add(res);
            }
        }
        content
    }
}

impl From<GenerateContentResponse> for Content {
    fn from(value: GenerateContentResponse) -> Self {
        let parts: Parts = value.candidates[0].content.parts().clone();
        Content::Model { parts }
    }
}

impl From<GenerateContentResponse> for Parts {
    fn from(value: GenerateContentResponse) -> Self {
        value.candidates[0].content.parts().clone()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ResponseCandidate {
    pub content: Content,
    pub finish_reason: String,
    pub index: u32,
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BlockReason {
    BlockReasonUnspecified,
    Safety,
    Other,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    pub block_reason: Option<BlockReason>,
    pub safety_ratings: Vec<SafetyRating>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    pub prompt_token_count: u32,
    pub candidates_token_count: Option<u32>,
    pub total_token_count: u32,
}

#[cfg(test)]
mod tests {
    use schemars::JsonSchema;
    use serde::Deserialize;
    use tokio_stream::StreamExt;

    use crate::{
        messages::{
            message::Content,
            tools::{HandlerError, Tool, ToolContext, Tools},
        },
        Gemini,
    };

    #[tokio::test]
    async fn test_generate_content_request() {
        let api_key = std::env::var("GEMINI_API_KEY").unwrap();
        let gemini = Gemini::builder().api_key(api_key).build().unwrap();
        let mut stream = gemini
            .generate_content()
            .add_content(Content::from("hello"))
            .model("gemini-1.5-flash")
            .build()
            .unwrap()
            .stream();

        while let Some(item) = stream.next().await {
            dbg!(&item.unwrap().candidates[0].content.parts()[0]
                .as_text()
                .unwrap());
        }
    }

    #[tokio::test]
    async fn test_function_calling() {
        #[derive(Debug, JsonSchema, Deserialize)]
        struct TestProps {
            #[schemars(description = "random number")]
            x: i32,
            #[schemars(description = "random number")]
            y: i32,
        }

        fn test_handler_1(props: TestProps, _cx: &ToolContext) -> Result<(), HandlerError> {
            println!("inside_handler 1 {:?}", props);
            Ok(())
        }
        fn test_handler_2(props: TestProps, _cx: &ToolContext) -> Result<(), HandlerError> {
            println!("inside_handler 2 {:?}", props);
            Ok(())
        }

        let api_key = std::env::var("GEMINI_API_KEY").unwrap();
        let gemini = Gemini::builder().api_key(api_key).build().unwrap();
        let tools = Tools::default()
            .with_tool(
                Tool::builder()
                    .name("test_function_1")
                    .description("use this to finish this test")
                    .handler(test_handler_1)
                    .build()
                    .unwrap(),
            )
            .with_tool(
                Tool::builder()
                    .name("test_function_2")
                    .description("use this to finish this test")
                    .handler(test_handler_2)
                    .build()
                    .unwrap(),
            );
        let mut res = gemini
            .generate_content()
            .add_content(Content::from(
                r#"to finish this test use both "test_function_1", "test_function_2" tools with random and different numbers"#,
            ))
            .model("gemini-1.5-flash")
            .tools(tools.clone())
            .build()
            .unwrap()
            .send()
            .await
            .unwrap();

        dbg!(&res);
        res.invoke_functions(&tools);
    }
}
