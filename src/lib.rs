pub mod files;
pub mod messages;

use core::fmt;

use messages::message::Content;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

const BASE_URL: &str = "https://generativelanguage.googleapis.com";

#[derive(Debug, PartialEq, PartialOrd, strum::EnumString, strum::Display)]
pub enum Model {
    #[strum(to_string = "gemini-1.5-flash")]
    Gemini15Flash,
    #[strum(to_string = "gemini-1.5-pro")]
    Gemini15Pro,
}

#[cfg(feature = "leaky-bucket")]
pub use leaky_bucket::RateLimiter;
#[cfg(feature = "leaky-bucket")]
use std::sync::Arc;

pub struct GeminiBuilder {
    api_key: Option<String>,
    client: Option<reqwest::Client>,
    #[cfg(feature = "leaky-bucket")]
    leaky_bucket: Option<RateLimiter>,
    api_version: String,
}

impl Default for GeminiBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            client: None,
            #[cfg(feature = "leaky-bucket")]
            leaky_bucket: None,
            api_version: String::from("v1beta"),
        }
    }
}

#[derive(Clone)]
pub struct Gemini {
    pub(crate) api_key: String,
    pub(crate) client: reqwest::Client,
    #[cfg(feature = "leaky-bucket")]
    pub(crate) leaky_bucket: Option<Arc<RateLimiter>>,
    pub(crate) api_version: String,
}

impl fmt::Debug for Gemini {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gemini")
            .field("api_key", &"[REDACTED]")
            .field("client", &self.client)
            .field("api_version", &self.api_version)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Error)]
pub enum ClientBuilderError {
    #[error("API key not set")]
    ApiKeyNotSet,
}

impl Gemini {
    #[must_use]
    pub fn builder() -> GeminiBuilder {
        GeminiBuilder::default()
    }
}

impl GeminiBuilder {
    #[must_use]
    pub fn with_api_key<T: Into<String>>(mut self, api_key: T) -> GeminiBuilder {
        self.api_key = Some(api_key.into());
        self
    }

    #[must_use]
    pub fn with_api_version<T: Into<String>>(mut self, api_version: T) -> GeminiBuilder {
        self.api_version = api_version.into();
        self
    }

    #[must_use]
    pub fn with_client(mut self, client: &reqwest::Client) -> GeminiBuilder {
        self.client = Some(client.clone());
        self
    }

    #[cfg(feature = "leaky-bucket")]
    #[must_use]
    pub fn with_limiter(mut self, leaky_bucket: RateLimiter) -> GeminiBuilder {
        self.leaky_bucket = Some(leaky_bucket);
        self
    }

    pub fn build(self) -> Result<Gemini, ClientBuilderError> {
        let Some(api_key) = self.api_key else {
            return Err(ClientBuilderError::ApiKeyNotSet);
        };

        let client = self.client.unwrap_or_default();

        #[cfg(feature = "leaky-bucket")]
        let leaky_bucket = self.leaky_bucket.map(Arc::new);

        Ok(Gemini {
            api_key,
            api_version: self.api_version,
            client,
            #[cfg(feature = "leaky-bucket")]
            leaky_bucket,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetySettings(Vec<SafetySetting>);

impl Default for SafetySettings {
    fn default() -> Self {
        Self(Vec::default())
            // .category(
            //     HarmCategory::HarmCategoryUnspecified,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategoryDerogatory,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategoryToxicity,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategoryViolence,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategorySexual,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategoryMedical,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategoryDangerous,
            //     HarmBlockThreshold::BlockNone,
            // )
            .with_category(
                HarmCategory::HarmCategoryHarassment,
                HarmBlockThreshold::BlockNone,
            )
            .with_category(
                HarmCategory::HarmCategoryHateSpeech,
                HarmBlockThreshold::BlockNone,
            )
            .with_category(
                HarmCategory::HarmCategorySexuallyExplicit,
                HarmBlockThreshold::BlockNone,
            )
            .with_category(
                HarmCategory::HarmCategoryDangerousContent,
                HarmBlockThreshold::BlockNone,
            )
    }
}

impl SafetySettings {
    #[must_use]
    pub fn with_category(mut self, category: HarmCategory, threshold: HarmBlockThreshold) -> Self {
        self.0.push((category, threshold).into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetySetting {
    pub category: HarmCategory,
    pub threshold: HarmBlockThreshold,
}

impl From<(HarmCategory, HarmBlockThreshold)> for SafetySetting {
    fn from(value: (HarmCategory, HarmBlockThreshold)) -> Self {
        SafetySetting {
            category: value.0,
            threshold: value.1,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmCategory {
    HarmCategoryUnspecified,
    HarmCategoryDerogatory,
    HarmCategoryToxicity,
    HarmCategoryViolence,
    HarmCategorySexual,
    HarmCategoryMedical,
    HarmCategoryDangerous,
    HarmCategoryHarassment,
    HarmCategoryHateSpeech,
    HarmCategorySexuallyExplicit,
    HarmCategoryDangerousContent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmBlockThreshold {
    HarmBlockThresholdUnspecified,
    BlockLowAndAbove,
    BlockMediumAndAbove,
    BlockOnlyHigh,
    BlockNone,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {
    pub category: HarmCategory,
    pub probability: String,
}

/// GenerationConfig
/// Configuration options for model generation and outputs. Not all parameters may be configurable for every model.
/// JSON representation
/// ```json
/// {
///   "stopSequences": [
///     "string"
///   ],
///   "responseMimeType": "string",
///   "responseSchema": {
///     "object": "Schema"
///   },
///   "candidateCount": 0,
///   "maxOutputTokens": 0,
///   "temperature": 0.0,
///   "topP": 0.0,
///   "topK": 0
/// }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop at the first appearance of a stop sequence. The stop sequence will not be included as part of the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Output response mimetype of the generated candidate text. Supported mimetype: text/plain: (default) Text output. application/json: JSON response in the candidates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// Output response schema of the generated candidate text when response mime type can have schema. Schema can be objects, primitives or arrays and is a subset of OpenAPI schema.
    /// If set, a compatible responseMimeType must also be set. Compatible mimetypes: application/json: Schema for JSON response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
    /// Number of generated responses to return.
    /// Currently, this value can only be set to 1. If unset, this will default to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<i32>,
    /// The maximum number of tokens to include in a candidate.
    /// Note: The default value varies by model, see the Model.output_token_limit attribute of the Model returned from the getModel function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    /// Controls the randomness of the output.
    /// Note: The default value varies by model, see the Model.temperature attribute of the Model returned from the getModel function.
    /// Values can range from [0.0, 2.0].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// The maximum cumulative probability of tokens to consider when sampling.
    /// The model uses combined Top-k and nucleus sampling.
    /// Tokens are sorted based on their assigned probabilities so that only the most likely tokens are considered. Top-k sampling directly limits the maximum number of tokens to consider, while Nucleus sampling limits number of tokens based on the cumulative probability.
    /// Note: The default value varies by model, see the Model.top_p attribute of the Model returned from the getModel function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// The maximum number of tokens to consider when sampling.
    /// Models use nucleus sampling or combined Top-k and nucleus sampling. Top-k sampling considers the set of topK most probable tokens. Models running with nucleus sampling don't allow topK setting.
    /// Note: The default value varies by model, see the Model.top_k attribute of the Model returned from the getModel function. Empty topK field in Model indicates the model doesn't apply top-k sampling and doesn't allow setting topK on requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
}

impl GenerationConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    #[must_use]
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }
    #[must_use]
    pub fn with_response_mime_type(mut self, response_mime_type: String) -> Self {
        self.response_mime_type = Some(response_mime_type);
        self
    }
    #[must_use]
    pub fn with_response_schema<T: JsonSchema>(mut self) -> Self {
        let settings = schemars::gen::SchemaSettings::openapi3().with(|s| {
            s.inline_subschemas = true;
            s.meta_schema = None;
        });
        let gen = schemars::gen::SchemaGenerator::new(settings);
        let root_schema = gen.into_root_schema_for::<T>();
        let mut json_schema = serde_json::to_value(root_schema).unwrap();
        json_schema
            .as_object_mut()
            .unwrap()
            .remove("title")
            .unwrap();
        self.response_schema = Some(json_schema);
        self
    }
    #[must_use]
    pub fn with_candidate_count(mut self, candidate_count: i32) -> Self {
        self.candidate_count = Some(candidate_count);
        self
    }
    #[must_use]
    pub fn with_max_output_tokens(mut self, max_output_tokens: i32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
    #[must_use]
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

#[derive(Debug, Deserialize, thiserror::Error)]
#[error("{error}")]
struct ErrorResponse {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorDetail {
    message: String,
    #[serde(default)]
    param: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

impl fmt::Display for ApiErrorDetail {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(param) = &self.param {
            write!(f, " (parameter: {param})")?;
        }
        if let Some(code) = &self.code {
            write!(f, " (code: {code})")?;
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ApiRequestError {
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
    #[error("Unexpected response from API: {response}")]
    UnexpectedResponse { response: String },
    #[error("Invalid event data: {0}")]
    InvalidEventData(String),
    #[error("Rate limit exceeded")]
    RateLimit,
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}
