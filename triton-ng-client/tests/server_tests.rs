#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::sut::Sut;
use anyhow::Result;
use std::collections::HashMap;

#[tokio::test]
async fn server_is_live() -> Result<()> {
    let config = Config::isolated("server_is_live", "mnist", &["mnist"]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let live = sut.server_live().await?;
    let actual = live;
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn server_is_ready() -> Result<()> {
    let config = Config::isolated("server_is_ready", "mnist", &["mnist"]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let ready = sut.server_ready().await?;
    let actual = ready;
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn server_metadata_contains_triton_name() -> Result<()> {
    let config =
        Config::isolated("server_metadata_contains_triton_name", "mnist", &["mnist"]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let info = sut.server_metadata().await?;
    let expected = ("triton".to_string(), true, true);
    let actual = (
        info.name,
        !info.version.is_empty(),
        !info.extensions.is_empty(),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn log_settings_are_readable() -> Result<()> {
    let config = Config::isolated("log_settings_are_readable", "mnist", &["mnist"]).await?;
    let sut = Sut::new(&config.triton_url).await?;
    let settings = sut.log_settings(HashMap::new()).await?;
    let actual = !settings.is_empty();
    let expected = true;
    assert_eq!(actual, expected);
    Ok(())
}
