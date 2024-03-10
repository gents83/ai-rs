use std::path::PathBuf;

use hf_hub::api::sync::ApiRepo;

pub mod token;

pub fn hf_retrieve(repo: &ApiRepo, file: &str) -> PathBuf {
    let mut path = PathBuf::new();
    let mut is_ok = false;
    while !is_ok {
        let r = repo.get(file);
        if let Ok(filename) = r {
            is_ok = true;
            path = filename;
        }
    }
    path
}

/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Option<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).unwrap();
    let json_file = std::fs::File::open(json_file).unwrap();
    let json: serde_json::Value = serde_json::from_reader(&json_file).unwrap();
    if let Some(weight_map) = match json.get("weight_map") {
        None => {
            println!(" weight map in {json_file:?}");
            None
        }
        Some(serde_json::Value::Object(map)) => Some(map),
        Some(_) => {
            println!(" weight map in {json_file:?} is not a map");
            None
        }
    } {
        let mut safetensors_files = std::collections::HashSet::new();
        for value in weight_map.values() {
            if let Some(file) = value.as_str() {
                safetensors_files.insert(file.to_string());
            }
        }
        let safetensors_files = safetensors_files
            .iter()
            .map(|v| hf_retrieve(repo, v))
            .collect::<Vec<_>>();
        Some(safetensors_files)
    } else {
        None
    }
}
