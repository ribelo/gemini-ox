use futures::{Stream, StreamExt};
use message::{Content, Contents, FunctionCall, Parts};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tools::{Tool, ToolBox, ToolConfig};

use crate::{ApiRequestError, Gemini, GenerationConfig, SafetyRating, SafetySettings, BASE_URL};

pub mod message;
pub mod tools;

#[derive(Debug, Serialize)]
pub struct GenerateContentRequest {
    contents: Contents,
    #[serde(skip_serializing_if = "ToolBox::is_empty")]
    tools: ToolBox,
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
    tools: Option<ToolBox>,
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
        let messages = self.contents.take().unwrap_or_default();
        self.contents = Some(messages.with_content(content));
        self
    }

    #[must_use]
    pub fn with_model<T: Into<String>>(mut self, model: T) -> Self {
        self.model = Some(model.into());
        self
    }

    #[must_use]
    pub fn with_tools<T: Into<ToolBox>>(mut self, tools: T) -> Self {
        self.tools = Some(tools.into());
        self
    }

    #[must_use]
    pub fn with_tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        let tools = self.tools.take().unwrap_or_default();
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
        self.contents = std::mem::take(&mut self.contents).with_content(content.into());
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
    pub async fn invoke_functions(&self, tools: &ToolBox) -> Option<Content> {
        let function_calls = self.get_function_calls();
        if function_calls.is_empty() {
            return None;
        }

        let mut content = Content::user();
        for fc in function_calls {
            let result = tools.invoke(fc.to_owned()).await;
            content.add_part(result);
        }

        (!content.is_empty()).then_some(content)
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
    use std::sync::{atomic::AtomicBool, Arc};

    use super::*;
    use async_trait::async_trait;
    use schemars::JsonSchema;
    use serde::Deserialize;

    use serde_json::json;
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

    #[derive(Clone)]
    pub struct ToolOne;
    #[derive(Serialize, Deserialize, JsonSchema)]
    pub struct ToolOneParams {
        number: i32,
    }

    #[async_trait]
    impl Tool for ToolOne {
        type Input = ToolOneParams;

        type Output = String;

        type Error = String;

        fn name(&self) -> String {
            "tool_one".to_string()
        }

        async fn invoke(&self, _input: Self::Input) -> Result<Self::Output, Self::Error> {
            Ok(self.name())
        }
    }

    #[derive(Clone)]
    pub struct ToolTwo;

    #[async_trait]
    impl Tool for ToolTwo {
        type Input = Value;

        type Output = String;

        type Error = String;

        fn name(&self) -> String {
            "tool_two".to_string()
        }

        async fn invoke(&self, _input: Self::Input) -> Result<Self::Output, Self::Error> {
            Ok(self.name())
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_function_calling() {
        let api_key = get_api_key();
        let gemini = Gemini::builder().with_api_key(api_key).build().unwrap();
        let tools = ToolBox::default();
        tools.add(ToolOne);
        tools.add(ToolTwo);

        let res = gemini
            .generate_content()
            .with_content(Content::from(
                r#"to finish this test use both "tool_one", "tool_two" tools"#,
            ))
            .with_model("gemini-1.5-flash")
            .with_tools(tools.clone())
            .build()
            .unwrap()
            .send()
            .await
            .unwrap();

        dbg!(&res);
        let content = res.invoke_functions(&tools).await;
        dbg!(content);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_json_output() {
        #[derive(Debug, Serialize, Deserialize, JsonSchema)]
        pub struct Book {
            #[schemars(description = "Unique identifier for the book")]
            pub id: i64,

            #[schemars(description = "Title of the book")]
            pub title: String,

            #[schemars(description = "Author of the book")]
            pub author: String,

            #[schemars(description = "ISBN (International Standard Book Number)")]
            pub isbn: String,

            #[schemars(description = "Genre of the book")]
            pub genre: String,

            #[schemars(description = "Number of pages in the book")]
            #[schemars(range(min = 1))]
            pub page_count: i32,

            #[schemars(description = "Average rating of the book")]
            #[schemars(range(min = 0.0, max = 5.0))]
            pub rating: f32,

            #[schemars(description = "Whether the book is currently available")]
            pub available: bool,

            #[schemars(description = "Tags associated with the book")]
            #[schemars(length(max = 5))]
            pub tags: Vec<String>,
        }

        let api_key = get_api_key();
        let gemini = Gemini::builder().with_api_key(api_key).build().unwrap();
        let config = GenerationConfig::default()
            .with_response_mime_type("application/json".to_string())
            .with_response_schema::<Book>();

        let res = gemini
            .generate_content()
            .with_content(Content::from(r"Describe Peter Watts Echopraxia book"))
            .with_model("gemini-1.5-flash")
            .with_generation_config(config)
            .build()
            .unwrap()
            .send()
            .await
            .unwrap();

        let text = res.candidates[0].content.parts()[0]
            .as_text()
            .unwrap()
            .to_string();
        let json = serde_json::from_str::<Book>(&text).unwrap();

        dbg!(&json);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_messages_request_success() {
        #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
        struct TestHandlerProps {
            random_number: i32,
        }

        #[derive(Clone, Debug, Default)]
        struct TestTool {
            is_called: Arc<AtomicBool>,
        }

        #[async_trait]
        impl Tool for TestTool {
            type Input = TestHandlerProps;

            type Output = Value;

            type Error = String;

            fn name(&self) -> String {
                "test_tool".to_string()
            }

            async fn invoke(&self, _input: Self::Input) -> Result<Self::Output, Self::Error> {
                self.is_called
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(json!({"result": "To finish this test use [finish_test] tool"}))
            }
        }

        #[derive(Default, Debug, Clone)]
        struct FinishTool {
            is_called: Arc<AtomicBool>,
        }

        #[async_trait]
        impl Tool for FinishTool {
            type Input = Value;

            type Output = Value;

            type Error = String;

            fn name(&self) -> String {
                "finish_tool".to_string()
            }

            async fn invoke(&self, _input: Self::Input) -> Result<Self::Output, Self::Error> {
                self.is_called
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(json!({"result": "Congratulations! You finished the test." }))
            }
        }

        let api_key = get_api_key();
        let gemini = Gemini::builder().with_api_key(api_key).build().unwrap();

        let tools = ToolBox::default();
        let test_tool = TestTool::default();
        tools.add(test_tool.clone());
        let finish_tool = FinishTool::default();
        tools.add(finish_tool.clone());

        let mut contents = Contents::default().with_content(Content::from(
            "This is testing environent. To continue this test use [test_tool]",
        ));

        let mut i = 0;
        loop {
            if i > 3 {
                dbg!(contents);
                dbg!(test_tool);
                dbg!(finish_tool);
                panic!("Test failed");
            }
            i += 1;

            let res = match gemini
                .generate_content()
                .with_contents(contents.clone())
                .with_model("gemini-1.5-flash")
                .with_tools(tools.clone())
                .build()
                .unwrap()
                .send()
                .await
            {
                Ok(res) => res,
                Err(err) => {
                    println!(
                        "---\n\n{}\n\n---",
                        serde_json::to_string_pretty(&contents).unwrap()
                    );
                    dbg!(test_tool);
                    dbg!(finish_tool);
                    panic!("{err}")
                }
            };

            contents.add_content(res.clone());

            let tool_results = res.invoke_functions(&tools).await;
            match tool_results {
                None => {
                    contents.add_content(Content::from("Follow instructions and use tools!"));
                }
                Some(results) => {
                    contents.add_content(results);
                }
            }
            println!("{}", serde_json::to_string_pretty(&contents).unwrap());

            if finish_tool
                .is_called
                .load(std::sync::atomic::Ordering::Relaxed)
                && test_tool
                    .is_called
                    .load(std::sync::atomic::Ordering::Relaxed)
            {
                #[cfg(target_arch = "wasm32")]
                console_log!("Test passed");
                println!("Test passed");
                println!(
                    "---\n\n{}\n\n---",
                    serde_json::to_string_pretty(&contents).unwrap()
                );
                break;
            }
        }
    }
}
