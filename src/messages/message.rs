use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents the role of a message sender in a conversation.
///
/// This enum distinguishes between messages sent by the user and
/// messages generated by the language model.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Model,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Content {
    User { parts: Parts },
    Model { parts: Parts },
}

impl Content {
    #[must_use]
    pub fn user() -> Self {
        Self::User {
            parts: Parts::default(),
        }
    }

    #[must_use]
    pub fn model() -> Self {
        Self::Model {
            parts: Parts::default(),
        }
    }

    #[must_use]
    pub fn parts(&self) -> &Parts {
        match self {
            Content::Model { parts: p } | Content::User { parts: p } => p,
        }
    }

    #[must_use]
    pub fn set_parts<P, I>(mut self, parts: I) -> Self
    where
        P: Into<Part>,
        I: IntoIterator<Item = P>,
    {
        match &mut self {
            Content::Model { parts: p } | Content::User { parts: p } => {
                *p = parts.into_iter().map(Into::into).collect();
            }
        }
        self
    }

    pub fn push_part<T: Into<Part>>(&mut self, part: T) {
        match self {
            Content::Model { parts } | Content::User { parts } => parts.push(part.into()),
        }
    }

    #[must_use]
    pub fn add_part<T: Into<Part>>(mut self, part: T) -> Self {
        self.push_part(part);
        self
    }

    #[must_use]
    pub fn as_user(&self) -> Option<&Self> {
        match self {
            Content::User { .. } => Some(self),
            Content::Model { .. } => None,
        }
    }

    #[must_use]
    pub fn expect_user(&self) -> &Self {
        self.as_user().expect("Expected Content to be User")
    }

    #[must_use]
    pub fn as_model(&self) -> Option<&Self> {
        match self {
            Content::Model { .. } => Some(self),
            Content::User { .. } => None,
        }
    }

    #[must_use]
    pub fn expect_model(&self) -> &Self {
        self.as_model().expect("Expected Content to be Model")
    }
}

impl FromIterator<Part> for Content {
    fn from_iter<T: IntoIterator<Item = Part>>(iter: T) -> Self {
        Self::user().set_parts(iter)
    }
}

impl From<&'static str> for Content {
    fn from(value: &'static str) -> Self {
        Self::user().add_part(value)
    }
}

impl Extend<Part> for Content {
    fn extend<T: IntoIterator<Item = Part>>(&mut self, iter: T) {
        match self {
            Content::User { parts } | Content::Model { parts } => parts.extend(iter),
        }
    }
}

/// Part
///
/// A datatype containing media that is part of a multi-part Content message.
/// A Part consists of data which has an associated datatype. A Part can only contain one of the accepted types in Part.data.
/// A Part must have a fixed IANA MIME type identifying the type and subtype of the media if the inlineData field is filled with raw bytes.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Part {
    /// Inline text.
    Text(Text),
    /// Inline media bytes.
    InlineData(Blob),
    /// A predicted FunctionCall returned from the model that contains a string representing the FunctionDeclaration.name with the arguments and their values.
    FunctionCall(FunctionCall),
    /// The result output of a FunctionCall that contains a string representing the FunctionDeclaration.name and a structured JSON object containing any output from the function is used as context to the model.
    FunctionResponse(FunctionResponse),
    /// URI based data.
    FileData(FileData),
}

impl Part {
    /// If the `Part` is a `Text` variant, return `Some(Text)`, otherwise return `None`.
    #[must_use]
    pub fn as_text(&self) -> Option<&Text> {
        match self {
            Part::Text(text) => Some(text),
            _ => None,
        }
    }

    #[must_use]
    pub fn expect_text(&self) -> &Text {
        self.as_text().expect("Expected Part to be Text")
    }
    /// If the `Part` is a `InlineData` variant, return `Some(InlineData)`, otherwise return `None`.
    #[must_use]
    pub fn as_inline_data(&self) -> Option<&Blob> {
        match self {
            Part::InlineData(inline_data) => Some(inline_data),
            _ => None,
        }
    }
    #[must_use]
    pub fn expect_inline_data(&self) -> &Blob {
        self.as_inline_data()
            .expect("Expected Part to be InlineData")
    }
    /// If the `Part` is a `FunctionCall` variant, return `Some(FunctionCall)`, otherwise return `None`.
    #[must_use]
    pub fn as_function_call(&self) -> Option<&FunctionCall> {
        match self {
            Part::FunctionCall(function_call) => Some(function_call),
            _ => None,
        }
    }
    #[must_use]
    pub fn expect_function_call(&self) -> &FunctionCall {
        self.as_function_call()
            .expect("Expected Part to be FunctionCall")
    }
    /// If the `Part` is a `FunctionResponse` variant, return `Some(FunctionResponse)`, otherwise return `None`.
    #[must_use]
    pub fn as_function_response(&self) -> Option<&FunctionResponse> {
        match self {
            Part::FunctionResponse(function_response) => Some(function_response),
            _ => None,
        }
    }
    #[must_use]
    pub fn expect_function_response(&self) -> &FunctionResponse {
        self.as_function_response()
            .expect("Expected Part to be FunctionResponse")
    }
    /// If the `Part` is a `FileData` variant, return `Some(FileData)`, otherwise return `None`.
    #[must_use]
    pub fn as_file_data(&self) -> Option<&FileData> {
        match self {
            Part::FileData(file_data) => Some(file_data),
            _ => None,
        }
    }
    #[must_use]
    pub fn expect_file_data(&self) -> &FileData {
        self.as_file_data().expect("Expected Part to be FileData")
    }
}

impl From<String> for Part {
    fn from(value: String) -> Self {
        Part::Text(value.into())
    }
}

impl From<&'static str> for Part {
    fn from(value: &'static str) -> Self {
        Part::Text(value.into())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Text(pub String);

impl From<String> for Text {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&'static str> for Text {
    fn from(value: &'static str) -> Self {
        Self(value.to_string())
    }
}

impl fmt::Display for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.0)
    }
}

impl From<Text> for Part {
    fn from(text: Text) -> Self {
        Self::Text(text)
    }
}

impl From<Blob> for Part {
    fn from(blob: Blob) -> Self {
        Self::InlineData(blob)
    }
}

impl From<FunctionCall> for Part {
    fn from(function_call: FunctionCall) -> Self {
        Self::FunctionCall(function_call)
    }
}

impl From<FunctionResponse> for Part {
    fn from(function_response: FunctionResponse) -> Self {
        Self::FunctionResponse(function_response)
    }
}

impl From<FileData> for Part {
    fn from(file_data: FileData) -> Self {
        Self::FileData(file_data)
    }
}

/// Blob
///
/// Raw media bytes.
/// Text should not be sent as raw bytes, use the 'text' field.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Blob {
    /// The IANA standard MIME type of the source data. Examples: - image/png - image/jpeg
    /// If an unsupported MIME type is provided, an error will be returned.
    /// For a complete list of supported types, see Supported file formats.
    pub mime_type: String,
    /// Raw bytes for media formats.
    /// A base64-encoded string.
    pub data: String,
}

/// FunctionCall
///
/// A predicted FunctionCall returned from the model that contains a string representing the FunctionDeclaration.name with the arguments and their values.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCall {
    /// Required. The name of the function to call. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 63.
    pub name: String,
    /// Optional. The function parameters and values in JSON object format.
    pub args: Option<Value>,
}

/// FunctionResponse
///
/// The result output from a FunctionCall that contains a string representing the FunctionDeclaration.name and a structured JSON object containing any output from the function is used as context to the model. This should contain the result of aFunctionCall made based on model prediction.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponse {
    /// Required. The name of the function to call. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 63.
    pub name: String,
    /// Required. The function response in JSON object format.
    pub response: Value,
}

/// FileData
///
/// URI based data.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct FileData {
    /// Optional. The IANA standard MIME type of the source data.
    pub mime_type: Option<String>,
    /// Required. URI.
    pub file_uri: String,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Parts(pub Vec<Part>);

impl Parts {
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&mut self, part: Part) {
        self.0.push(part);
    }

    pub fn get_function_calls(&self) -> Vec<&FunctionCall> {
        self.0
            .iter()
            .filter_map(|part| match part {
                Part::FunctionCall(function_call) => Some(function_call),
                _ => None,
            })
            .collect()
    }
}

impl FromIterator<Part> for Parts {
    fn from_iter<T: IntoIterator<Item = Part>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for Parts {
    type Item = Part;
    type IntoIter = <Vec<Part> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Extend<Part> for Parts {
    fn extend<T: IntoIterator<Item = Part>>(&mut self, iter: T) {
        // Parts is just a Vec<Part>, so extend it directly
        self.0.extend(iter);
    }
}

/// Allows accessing parts by index.
impl std::ops::Index<usize> for Parts {
    type Output = Part;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
/// Allows modifying parts by index.
impl std::ops::IndexMut<usize> for Parts {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Contents(pub Vec<Content>);

impl Contents {
    pub fn push<T: Into<Content>>(&mut self, message: T) {
        self.0.push(message.into());
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl From<Content> for Contents {
    fn from(value: Content) -> Self {
        Contents(vec![value])
    }
}

impl From<Vec<Content>> for Contents {
    fn from(value: Vec<Content>) -> Self {
        Contents(value)
    }
}

impl FromIterator<Content> for Contents {
    fn from_iter<T: IntoIterator<Item = Content>>(iter: T) -> Self {
        Contents(iter.into_iter().collect())
    }
}

impl<T> Extend<T> for Contents
where
    T: Into<Content>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.0.extend(iter.into_iter().map(Into::into));
    }
}

impl std::ops::Index<usize> for Contents {
    type Output = Content;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IntoIterator for Contents {
    type Item = Content;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_role_serialization() {
        let user_role = Role::User;
        let user_json = serde_json::to_string(&user_role).unwrap();
        assert_eq!(user_json, "\"user\"");

        let model_role = Role::Model;
        let model_json = serde_json::to_string(&model_role).unwrap();
        assert_eq!(model_json, "\"model\"");
    }

    #[test]
    fn test_role_deserialization() {
        let user_json = "\"user\"";
        let user_role: Role = serde_json::from_str(user_json).unwrap();
        assert_eq!(user_role, Role::User);

        let model_json = "\"model\"";
        let model_role: Role = serde_json::from_str(model_json).unwrap();
        assert_eq!(model_role, Role::Model);
    }

    #[test]
    fn test_content_serialization() {
        let content =
            Content::user().set_parts(vec![Part::Text(Text("Hello, world!".to_string()))]);
        let json_content = serde_json::to_string(&content).unwrap();
        assert_eq!(
            json_content,
            r#"{"role":"user","parts":[{"text":"Hello, world!"}]}"#
        );
    }

    #[test]
    fn test_content_deserialization() {
        let json_content = r#"{"parts":[{"text":"Hello, world!"}],"role":"user"}"#;
        let content: Content = serde_json::from_str(json_content).unwrap();
        assert_eq!(
            content,
            Content::user().set_parts(vec![Part::Text(Text("Hello, world!".to_string(),))])
        );
    }

    #[test]
    fn test_part_serialization() {
        let part = Part::Text(Text("Hello, world!".to_string()));
        let json_part = serde_json::to_string(&part).unwrap();
        assert_eq!(json_part, r#"{"text":"Hello, world!"}"#);
    }

    #[test]
    fn test_part_deserialization() {
        let json_part = r#"{"text":"Hello, world!"}"#;
        let part: Part = serde_json::from_str(json_part).unwrap();
        assert_eq!(part, Part::Text(Text("Hello, world!".to_string(),)));
    }

    #[test]
    fn test_text_serialization() {
        let text = Text("Hello, world!".to_string());
        let json_text = serde_json::to_string(&text).unwrap();
        assert_eq!(json_text, r#"{"text":"Hello, world!"}"#);
    }

    #[test]
    fn test_text_deserialization() {
        let json_text = r#"{"text":"Hello, world!"}"#;
        let text: Text = serde_json::from_str(json_text).unwrap();
        assert_eq!(text, Text("Hello, world!".to_string(),));
    }

    #[test]
    fn test_function_call_serialization() {
        let function_call = FunctionCall {
            name: "my_function".to_string(),
            args: Some(json!({"arg1": "value1", "arg2": 42})),
        };
        let json_function_call = serde_json::to_string(&function_call).unwrap();
        assert_eq!(
            json_function_call,
            r#"{"name":"my_function","args":{"arg1":"value1","arg2":42}}"#
        );
    }

    #[test]
    fn test_function_call_deserialization() {
        let json_function_call = r#"{"name":"my_function","args":{"arg1":"value1","arg2":42}}"#;
        let function_call: FunctionCall = serde_json::from_str(json_function_call).unwrap();
        assert_eq!(
            function_call,
            FunctionCall {
                name: "my_function".to_string(),
                args: Some(json!({"arg1": "value1", "arg2": 42})),
            }
        );
    }

    #[test]
    fn test_function_response_serialization() {
        let function_response = FunctionResponse {
            name: "my_function".to_string(),
            response: json!({"result": "success"}),
        };
        let json_function_response = serde_json::to_string(&function_response).unwrap();
        assert_eq!(
            json_function_response,
            r#"{"name":"my_function","response":{"result":"success"}}"#
        );
    }

    #[test]
    fn test_function_response_deserialization() {
        let json_function_response = r#"{"name":"my_function","response":{"result":"success"}}"#;
        let function_response: FunctionResponse =
            serde_json::from_str(json_function_response).unwrap();
        assert_eq!(
            function_response,
            FunctionResponse {
                name: "my_function".to_string(),
                response: json!({"result": "success"}),
            }
        );
    }
    #[test]
    fn test_deserialize_content_with_function_call() {
        let input = json!({
            "parts": [
            {
                "functionCall": {
                    "name": "test_function",
                    "args": {
                        "x": 1,
                        "y": 2
                    }
                }
            }
            ],
            "role": "model"
        });

        let expected = Content::Model {
            parts: Parts(vec![Part::FunctionCall(FunctionCall {
                name: "test_function".to_string(),
                args: Some(json!({
                    "x": 1,
                    "y": 2
                })),
            })]),
        };
        let result: Content = serde_json::from_value(input).unwrap();
        assert_eq!(result, expected);
    }
}
