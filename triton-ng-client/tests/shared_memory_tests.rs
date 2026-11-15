#[path = "common/mod.rs"]
mod common;

use crate::common::config::Config;
use crate::common::sut::{Sut, create_posix_shm, remove_posix_shm};
use anyhow::{Context, Result};

#[tokio::test]
async fn system_shared_memory_status_omits_an_unregistered_region() -> Result<()> {
    let config = Config::isolated(
        "system_shared_memory_status_omits_an_unregistered_region",
        "mnist",
        &[],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let name = config.model("mnist");
    let regions = sut.system_shared_memory_status(None).await?;
    let expected = false;
    let actual = regions.contains_key(name);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn cuda_shared_memory_status_omits_an_unregistered_region() -> Result<()> {
    let config = Config::isolated(
        "cuda_shared_memory_status_omits_an_unregistered_region",
        "mnist",
        &[],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let name = config.model("mnist");
    let regions = sut.cuda_shared_memory_status(None).await?;
    let expected = false;
    let actual = regions.contains_key(name);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn system_shared_memory_status_describes_a_registered_region() -> Result<()> {
    let config = Config::isolated(
        "system_shared_memory_status_describes_a_registered_region",
        "mnist",
        &[],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let name = config.model("mnist");
    let key = format!("/{name}");
    create_posix_shm(&key, 64)?;
    sut.system_shared_memory_register(name, &key, 0, 64).await?;
    let regions = sut.system_shared_memory_status(Some(name)).await?;
    let region = regions.get(name).context("registered region not found")?;
    let registered = (region.key.clone(), region.offset, region.byte_size);
    sut.system_shared_memory_unregister(Some(name)).await?;

    remove_posix_shm(&key)?;
    let expected = (key, 0, 64);
    let actual = registered;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn system_shared_memory_unregister_removes_its_region() -> Result<()> {
    let config = Config::isolated(
        "system_shared_memory_unregister_removes_its_region",
        "mnist",
        &[],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let name = config.model("mnist");
    let key = format!("/{name}");
    create_posix_shm(&key, 64)?;
    sut.system_shared_memory_register(name, &key, 0, 64).await?;
    let regions = sut.system_shared_memory_status(Some(name)).await?;
    let region = regions.get(name).context("registered region not found")?;
    let _ = region;
    sut.system_shared_memory_unregister(Some(name)).await?;
    let remaining = sut.system_shared_memory_status(None).await?;
    remove_posix_shm(&key)?;
    let expected = false;
    let actual = remaining.contains_key(name);
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn cuda_shared_memory_register_rejects_an_invalid_handle() -> Result<()> {
    let config = Config::isolated(
        "cuda_shared_memory_register_rejects_an_invalid_handle",
        "mnist",
        &[],
    )
    .await?;
    let sut = Sut::new(&config.triton_url).await?;
    let result = sut
        .cuda_shared_memory_register(config.model("mnist"), vec![0u8; 64], 0, 64)
        .await;
    let expected = true;
    let actual = result.is_err();
    assert_eq!(actual, expected);
    Ok(())
}
