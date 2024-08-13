use serde_json::json;
use std::{fs::File, io::Read, path::PathBuf};

use crate::{ApiRequestError, Gemini, BASE_URL};

pub struct FileUploadRequest {
    file_name: String,
    mime_type: String,
    data: Vec<u8>,
    gemini: Gemini,
}

pub struct FileUploadRequestBuilder {
    file_name: Option<String>,
    mime_type: Option<String>,
    data: Option<Vec<u8>>,
    path: Option<PathBuf>,
    gemini: Gemini,
}

impl FileUploadRequestBuilder {
    #[must_use]
    pub fn new(gemini: Gemini) -> Self {
        FileUploadRequestBuilder {
            file_name: None,
            mime_type: None,
            data: None,
            path: None,
            gemini,
        }
    }

    #[must_use]
    pub fn with_file_name<T: Into<String>>(mut self, file_name: T) -> Self {
        self.file_name = Some(file_name.into());
        self
    }

    #[must_use]
    pub fn with_mime_type<T: Into<String>>(mut self, mime_type: T) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    #[must_use]
    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.data = Some(data);
        self
    }

    #[must_use]
    pub fn with_path<T: Into<PathBuf>>(mut self, path: T) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn build(self) -> Result<FileUploadRequest, ApiRequestError> {
        let (file_name, mime_type, data) = if let Some(path) = self.path {
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| ApiRequestError::InvalidRequestError {
                    code: None,
                    details: json!({}),
                    message: "Invalid file name".to_string(),
                    status: None,
                })?
                .to_string();

            let mime_type = self.mime_type.unwrap_or_else(|| {
                mime_guess::from_path(&path)
                    .first_or_octet_stream()
                    .to_string()
            });

            let mut file = File::open(&path)?;
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;

            (file_name, mime_type, buf)
        } else if let Some(data) = self.data {
            let file_name = self
                .file_name
                .ok_or_else(|| ApiRequestError::InvalidRequestError {
                    code: None,
                    details: json!({}),
                    message: "File name is required when using raw data".to_string(),
                    status: None,
                })?;

            let mime_type = self.mime_type.unwrap_or_else(|| {
                mime_guess::from_path(&file_name)
                    .first_or_octet_stream()
                    .to_string()
            });

            (file_name, mime_type, data)
        } else {
            return Err(ApiRequestError::InvalidRequestError {
                code: None,
                details: json!({}),
                message: "Either path or data must be provided".to_string(),
                status: None,
            });
        };

        Ok(FileUploadRequest {
            file_name,
            mime_type,
            data,
            gemini: self.gemini,
        })
    }
}

impl FileUploadRequest {
    pub async fn send(&self) -> Result<String, ApiRequestError> {
        let num_bytes = self.data.len();

        let init_url = format!(
            "{}/upload/{}/files?key={}",
            BASE_URL, self.gemini.api_version, self.gemini.api_key
        );

        let init_response = self
            .gemini
            .client
            .post(&init_url)
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", num_bytes.to_string())
            .header("X-Goog-Upload-Header-Content-Type", &self.mime_type)
            .json(&json!({
                "file": {
                    "display_name": self.file_name
                }
            }))
            .send()
            .await?;

        let upload_url = init_response
            .headers()
            .get("X-Goog-Upload-URL")
            .and_then(|h| h.to_str().ok())
            .ok_or_else(|| ApiRequestError::InvalidRequestError {
                code: None,
                details: json!({}),
                message: "Missing upload URL in response".to_string(),
                status: None,
            })?
            .to_string();

        let upload_response = self
            .gemini
            .client
            .post(&upload_url)
            .header("Content-Length", num_bytes.to_string())
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .body(self.data.clone())
            .send()
            .await?;

        let file_info: serde_json::Value = upload_response.json().await?;
        let file_uri = file_info["file"]["uri"]
            .as_str()
            .ok_or_else(|| ApiRequestError::InvalidRequestError {
                code: None,
                details: json!({}),
                message: "Missing file URI in response".to_string(),
                status: None,
            })?
            .to_string();

        Ok(file_uri)
    }
}

impl Gemini {
    pub fn upload_file(&self) -> FileUploadRequestBuilder {
        FileUploadRequestBuilder::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_file_upload_request_send() {
        let api_key = get_api_key();
        let gemini = Gemini::builder()
            .with_api_key(api_key)
            .with_api_version("v1beta")
            .build()
            .expect("Failed to build Gemini client");

        let file_path = PathBuf::from("/home/ribelo/documents/kio/2009_1488.pdf");
        let request = gemini
            .upload_file()
            .with_path(&file_path)
            .build()
            .expect("Failed to build file upload request");

        let result = request.send().await;
        assert!(result.is_ok(), "File upload failed: {:?}", result.err());

        let file_uri = result.expect("Failed to get file URI");
        tracing::info!("Uploaded file URI: {}", file_uri);
        assert!(
            file_uri.starts_with("https://"),
            "File URI does not start with 'https://'"
        );
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test]
    async fn test_file_upload_request_send() {
        let api_key = get_api_key();
        let gemini = Gemini::builder()
            .with_api_key(api_key)
            .with_api_version("v1beta")
            .build()
            .expect("Failed to build Gemini client");

        let file_content = include_bytes!("/home/ribelo/documents/kio/2009_1488.pdf");
        let request = gemini
            .upload_file()
            .with_file_name("")
            .with_data(file_content.to_vec())
            .build()
            .expect("Failed to build file upload request");

        let result = request.send().await;
        assert!(result.is_ok(), "File upload failed: {:?}", result.err());

        let file_uri = result.expect("Failed to get file URI");
        console_log!("Uploaded file URI: {file_uri}");
        assert!(
            file_uri.starts_with("https://"),
            "File URI does not start with 'https://'"
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_file_upload_request_builder_with_data() {
        let api_key = get_api_key();
        let gemini = Gemini::builder()
            .with_api_key(api_key)
            .with_api_version("v1beta")
            .build()
            .unwrap();

        let data = b"Test data".to_vec();
        let request = gemini
            .upload_file()
            .with_file_name("test.txt")
            .with_mime_type("text/plain")
            .with_data(data.clone())
            .build()
            .unwrap();

        assert_eq!(request.file_name, "test.txt");
        assert_eq!(request.mime_type, "text/plain");
        assert_eq!(request.data, data);

        let result = request.send().await;
        assert!(result.is_ok(), "File upload failed: {:?}", result.err());

        let file_uri = result.unwrap();
        println!("Uploaded file URI: {}", file_uri);
        assert!(file_uri.starts_with("https://"));
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_file_upload_request_builder_missing_data() {
        let api_key = get_api_key();
        let gemini = Gemini::builder()
            .with_api_key(api_key)
            .with_api_version("v1beta")
            .build()
            .unwrap();

        let result = gemini.upload_file().with_file_name("test.txt").build();

        assert!(result.is_err());
    }
}
