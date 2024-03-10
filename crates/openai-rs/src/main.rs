use openai_api_rust::chat::*;
use openai_api_rust::*;

const OPEN_AI_API: &str = "https://api.openai.com/v1/";
const OPEN_AI_MODEL: &str = "gpt-3.5-turbo";
const OPEN_AI_MODEL_MAX_TOKENS: i32 = 4096;

fn main() {
    // Load API key from environment OPENAI_API_KEY.
    // You can also hadcode through `Auth::new(<your_api_key>)`, but it is not recommended.
    let auth = Auth::from_env().unwrap();
    let client = OpenAI::new(auth, OPEN_AI_API).use_env_proxy();
    let body = ChatBody {
        model: OPEN_AI_MODEL.to_string(),
        max_tokens: Some(OPEN_AI_MODEL_MAX_TOKENS),
        temperature: Some(1_f32),
        top_p: Some(0_f32),
        n: Some(1),
        stream: Some(false),
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
        messages: vec![Message {
            role: Role::User,
            content: "Hello!".to_string(),
        }],
    };
    let rs = client.chat_completion_create(&body);
    let choice = rs.unwrap().choices;
    let message = &choice[0].message.as_ref().unwrap();
    assert!(message.content.contains("Hello"));
}
