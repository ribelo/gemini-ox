use futures::{Stream, StreamExt};
use message::{Content, Contents, FunctionCall, Parts};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tools::{Tool, ToolConfig, Tools};

use crate::{ApiRequestError, Gemini, GenerationConfig, SafetyRating, SafetySettings, BASE_URL};

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

#[derive(Debug, thiserror::Error)]
pub enum GenerateContentRequestBuilderError {
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
    #[must_use]
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
    pub fn with_contents<T: Into<Contents>>(mut self, contents: T) -> Self {
        self.contents = Some(contents.into());
        self
    }

    #[must_use]
    pub fn with_content<T: Into<Content>>(mut self, content: T) -> Self {
        let mut messages = self.contents.take().unwrap_or_default();
        messages.push(content);
        self.contents = Some(messages);
        self
    }

    #[must_use]
    pub fn with_model<T: Into<String>>(mut self, model: T) -> Self {
        self.model = Some(model.into());
        self
    }

    #[must_use]
    pub fn with_tools<T: Into<Tools>>(mut self, tools: T) -> Self {
        self.tools = Some(tools.into());
        self
    }

    #[must_use]
    pub fn with_tool<T: Into<Tool>>(mut self, tool: T) -> Self {
        let mut tools = self.tools.take().unwrap_or_default();
        tools.add(tool);
        self.tools = Some(tools);
        self
    }

    #[must_use]
    pub fn with_tool_config<T: Into<ToolConfig>>(mut self, cfg: T) -> Self {
        self.tool_config = Some(cfg.into());
        self
    }

    #[must_use]
    pub fn with_safety_settings(mut self, safety_settings: SafetySettings) -> Self {
        self.safety_settings = safety_settings;
        self
    }

    #[must_use]
    pub fn with_system_instruction<T: Into<Content>>(mut self, system_instruction: T) -> Self {
        self.system_instruction = Some(system_instruction.into());
        self
    }

    #[must_use]
    pub fn with_generation_config(mut self, generation_config: GenerationConfig) -> Self {
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

impl Gemini {
    #[must_use]
    pub fn generate_content(&self) -> GenerateContentRequestBuilder {
        GenerateContentRequestBuilder::new(self.clone())
    }
}

impl GenerateContentRequest {
    pub async fn send(&self) -> Result<GenerateContentResponse, ApiRequestError> {
        let url = format!(
            "{}/{}/models/{}:generateContent?key={}",
            BASE_URL, self.gemini.api_version, self.model, self.gemini.api_key
        );
        let res = self.gemini.client.post(&url).json(self).send().await?;

        match res.status().as_u16() {
            200 | 201 => {
                let data: GenerateContentResponse = res.json().await?;
                Ok(data)
            }
            429 => Err(ApiRequestError::RateLimit),
            _ => {
                let mut e: Value = res.json().await?;
                Err(ApiRequestError::InvalidRequestError {
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

    pub async fn stream(
        &self,
    ) -> impl Stream<Item = Result<GenerateContentResponse, ApiRequestError>> {
        let url = format!(
            "{}/{}/models/{}:streamGenerateContent?alt=sse&key={}",
            BASE_URL, self.gemini.api_version, self.model, self.gemini.api_key
        );
        let stream = self
            .gemini
            .client
            .post(&url)
            .json(self)
            .send()
            .await
            .unwrap()
            .bytes_stream();

        stream.filter_map(|chunk| async move {
            match chunk {
                Ok(bytes) => {
                    let data = String::from_utf8(bytes.to_vec()).unwrap();
                    match data.as_str() {
                        "" => None,
                        s if s.starts_with("data: ") => {
                            let json_data = s.trim_start_matches("data: ");
                            Some(
                                serde_json::from_str::<GenerateContentResponse>(json_data)
                                    .map_err(ApiRequestError::SerdeError),
                            )
                        }
                        _ => Some(Err(ApiRequestError::InvalidEventData(data.to_string()))),
                    }
                }
                Err(e) => Some(Err(ApiRequestError::ReqwestError(e))),
            }
        })
    }

    pub fn add_content<T: Into<Content>>(&mut self, content: T) {
        self.contents.push(content.into());
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
    use super::*;
    use schemars::JsonSchema;
    use serde::Deserialize;

    use tools::{HandlerError, ToolContext};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    // #[cfg(target_arch = "wasm32")]
    // wasm_bindgen_test_configure!(run_in_browser);

    #[cfg(not(target_arch = "wasm32"))]
    fn get_api_key() -> String {
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set")
    }

    #[cfg(target_arch = "wasm32")]
    fn get_api_key() -> String {
        std::env!("GEMINI_API_KEY").to_string()
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_generate_content_request() {
        let api_key = get_api_key();
        let gemini = Gemini::builder().with_api_key(api_key).build().unwrap();
        let mut stream = Box::pin(
            gemini
                .generate_content()
                .with_content(Content::from("hello"))
                .with_model("gemini-1.5-flash")
                .build()
                .unwrap()
                .stream()
                .await,
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut responses = Vec::new();
            while let Some(Ok(item)) = stream.next().await {
                let content = item.candidates[0].content.parts()[0].clone();
                if let Some(text) = content.as_text() {
                    responses.push(text.to_string());
                }
            }
            assert!(!responses.is_empty());
        }

        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                let mut responses = Vec::new();
                while let Some(Ok(response)) = stream.next().await {
                    let content = &response.candidates[0].content.parts()[0];
                    if let Some(text) = content.as_text() {
                        responses.push(text.to_string());
                    }
                }
                assert!(!responses.is_empty());
            });
        }
    }

    #[derive(Debug, Clone, Copy, JsonSchema, Deserialize)]
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

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_function_calling() {
        let api_key = get_api_key();
        let gemini = Gemini::builder().with_api_key(api_key).build().unwrap();
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

        let res = gemini
            .generate_content()
            .with_content(Content::from(
                r#"to finish this test use both "test_function_1", "test_function_2" tools with random and different numbers"#,
            ))
            .with_model("gemini-1.5-flash")
            .with_tools(tools.clone())
            .build()
            .unwrap()
            .send()
            .await
            .unwrap();

        println!("Response: {:?}", res);
        res.invoke_functions(&tools);
    }
}
