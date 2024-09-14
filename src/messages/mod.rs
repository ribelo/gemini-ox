use futures::{Stream, StreamExt};
use message::{Content, FunctionCall, Part};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tools::ToolBox;
use typed_builder::TypedBuilder;

use crate::{ApiRequestError, Gemini, GenerationConfig, SafetyRating, SafetySettings, BASE_URL};

pub mod message;
pub mod tools;

#[derive(Debug, Serialize, TypedBuilder)]
pub struct GenerateContentRequest<'a, 'b> {
    #[builder(default, setter(transform = |v: impl IntoIterator<Item = impl Into<Content<'a>>>|
        v.into_iter().map(Into::into).collect::<Vec<_>>()
    ))]
    contents: Vec<Content<'a>>,
    #[builder(default)]
    #[serde(skip_serializing_if = "ToolBox::is_empty")]
    tools: ToolBox,
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_settings: Option<SafetySettings>,
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content<'b>>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[builder(setter(into))]
    model: String,
    #[serde(skip)]
    gemini: Gemini,
}

impl Gemini {
    pub fn generate_content(
        &self,
    ) -> GenerateContentRequestBuilder<'_, '_, ((), (), (), (), (), (), (Gemini,))> {
        GenerateContentRequest::builder().gemini(self.clone())
    }
}

impl<'a, 'b> GenerateContentRequest<'a, 'b> {
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
    ) -> impl Stream<Item = Result<GenerateContentResponse<'static>, ApiRequestError>> {
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

    pub fn add_content<T: Into<Content<'a>>>(&mut self, content: T) {
        self.contents.push(content.into());
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse<'a> {
    pub candidates: Vec<ResponseCandidate<'a>>,
    pub prompt_feedback: Option<PromptFeedback>,
    pub usage_metadata: Option<UsageMetadata>,
}

impl<'a> GenerateContentResponse<'a> {
    #[must_use]
    pub fn content(&'a self) -> Option<&'a Content<'a>> {
        self.candidates.first().map(|c| &c.content)
    }

    #[must_use]
    pub fn get_function_calls(&self) -> Vec<&FunctionCall> {
        self.content()
            .map(|c| {
                c.parts()
                    .iter()
                    .filter_map(|p| p.as_function_call())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[must_use]
    pub async fn invoke_functions(&'a self, tools: &ToolBox) -> Option<Content<'static>> {
        let function_calls = self.get_function_calls();
        if function_calls.is_empty() {
            return None;
        }

        let mut content = Content::builder().role(message::Role::User).build();
        for fc in function_calls {
            let result = tools.invoke(fc.clone()).await;
            content.push(result);
        }

        (!content.is_empty()).then_some(content)
    }

    pub fn to_owned(&self) -> GenerateContentResponse<'static> {
        GenerateContentResponse {
            candidates: self
                .candidates
                .iter()
                .map(ResponseCandidate::to_owned)
                .collect(),
            prompt_feedback: self.prompt_feedback.clone(),
            usage_metadata: self.usage_metadata.clone(),
        }
    }
}

impl<'a> From<GenerateContentResponse<'a>> for Content<'static> {
    fn from(value: GenerateContentResponse<'a>) -> Self {
        let parts = value.candidates[0].content.parts().clone();
        Content::builder()
            .role(message::Role::Model)
            .parts(parts)
            .build()
            .to_owned()
    }
}

impl<'a> From<GenerateContentResponse<'a>> for Vec<Part<'static>> {
    fn from(value: GenerateContentResponse<'a>) -> Self {
        value.candidates[0]
            .content
            .parts()
            .iter()
            .map(message::Part::to_owned)
            .collect()
    }
}

#[derive(
    Serialize, Deserialize, Debug, Clone, PartialEq, Eq, strum::EnumString, strum::Display,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    FinishReasonUnspecified,
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Language,
    Other,
    Blocklist,
    ProhibitedContent,
    Spii,
    MalformedFunctionCall,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ResponseCandidate<'a> {
    pub content: Content<'a>,
    pub finish_reason: FinishReason,
    pub index: u32,
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

impl<'a> ResponseCandidate<'a> {
    #[must_use]
    pub fn to_owned(&self) -> ResponseCandidate<'static> {
        ResponseCandidate {
            content: self.content.to_owned(),
            finish_reason: self.finish_reason.clone(),
            index: self.index,
            safety_ratings: self.safety_ratings.clone(),
        }
    }
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

    use crate::ResponseSchema;

    use super::*;
    use async_trait::async_trait;
    use futures::pin_mut;
    use schemars::JsonSchema;
    use serde::Deserialize;

    use serde_json::json;
    use tools::Tool;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    // #[cfg(target_arch = "wasm32")]
    // wasm_bindgen_test_configure!(run_in_browser);

    #[cfg(not(target_arch = "wasm32"))]
    fn get_api_key() -> String {
        std::env::var("GOOGLE_AI_API_KEY").expect("GOOGLE_AI_API_KEY must be set")
    }

    #[cfg(target_arch = "wasm32")]
    fn get_api_key() -> String {
        std::env!("GOOGLE_AI_API_KEY").to_string()
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_generate_content_request() {
        let api_key = get_api_key();
        let gemini = Gemini::builder().api_key(api_key).build();
        let request = gemini
            .generate_content()
            .contents(vec![Content::from("hello")])
            .model("gemini-1.5-flash")
            .build();

        let stream = request.stream().await;

        pin_mut!(stream);

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
        let gemini = Gemini::builder().api_key(api_key).build();
        let tools = ToolBox::default();
        tools.add(ToolOne);
        tools.add(ToolTwo);

        let request = gemini
            .generate_content()
            .contents(vec![Content::from(
                r#"to finish this test use both "tool_one", "tool_two" tools"#,
            )])
            .model("gemini-1.5-flash")
            .tools(tools.clone())
            .build();

        let response = request.send().await.unwrap();

        dbg!(&request);
        let content = response.invoke_functions(&tools).await;
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
        let gemini = Gemini::builder().api_key(api_key).build();
        dbg!(ResponseSchema::from::<Book>());
        let config = GenerationConfig::builder()
            .response_mime_type("application/json".to_string())
            .response_schema(ResponseSchema::from::<Book>())
            .build();

        let request = gemini
            .generate_content()
            .contents(vec![r"Describe Peter Watts Echopraxia book"])
            .model("gemini-1.5-flash")
            .generation_config(config)
            .build();

        let response = request.send().await.unwrap();

        let text = response.candidates[0].content.parts()[0]
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
        let gemini = Gemini::builder().api_key(api_key).build();

        let tools = ToolBox::default();
        let test_tool = TestTool::default();
        tools.add(test_tool.clone());
        let finish_tool = FinishTool::default();
        tools.add(finish_tool.clone());

        let mut contents = vec![Content::from(
            "This is testing environent. To continue this test use [test_tool]",
        )];

        let mut i = 0;
        loop {
            if i > 3 {
                dbg!(&contents);
                dbg!(&test_tool);
                dbg!(&finish_tool);
                panic!("Test failed");
            }
            i += 1;

            let request = gemini
                .generate_content()
                .contents(contents.clone())
                .model("gemini-1.5-flash")
                .tools(tools.clone())
                .build();

            let response = match request.send().await {
                Ok(res) => res,
                Err(err) => {
                    println!(
                        "---\n\n{}\n\n---",
                        serde_json::to_string_pretty(&contents).unwrap()
                    );
                    dbg!(&test_tool);
                    dbg!(&finish_tool);
                    panic!("{err}")
                }
            };

            contents.push(response.clone().into());

            let tool_results = response.invoke_functions(&tools).await;
            match tool_results {
                None => {
                    contents.push(Content::from("Follow instructions and use tools!"));
                }
                Some(results) => {
                    contents.push(results);
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
