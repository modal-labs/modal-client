use crate::error;
use std::{collections, env, fs, path};

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub active: Option<bool>,
    pub server_url: Option<String>,
    pub token_id: Option<String>,
    pub token_secret: Option<String>,
    pub task_id: Option<String>,
    pub task_secret: Option<String>,
}

type Configs = collections::HashMap<String, Config>;

impl Config {
    pub fn load() -> Result<Self, error::Error> {
        let home_dir =
            path::PathBuf::from(env::var_os("HOME").ok_or(error::Error::HomeEnvVarNotSet)?);
        let contents = fs::read_to_string(home_dir.join(".modal.toml"))?;
        let configs: Configs = toml::from_str(&contents)?;

        for config in configs.into_values() {
            if config.active.unwrap_or_default() {
                return Ok(config);
            }
        }
        Err(error::Error::NoActiveConfigFound)
    }
}
