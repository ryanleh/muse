use io_utils::{IMuxAsync, IMuxSync};

#[inline]
pub fn serialize<W, T>(w: &mut IMuxSync<W>, value: &T) -> Result<(), bincode::Error>
where
    W: std::io::Write + Send,
    T: serde::Serialize + ?Sized,
{
    let bytes: Vec<u8> = bincode::serialize(value)?;
    let _ = w.write(&bytes)?;
    Ok(())
}

#[inline]
pub fn deserialize<R, T>(reader: &mut IMuxSync<R>) -> bincode::Result<T>
where
    R: std::io::Read + Send,
    T: serde::de::DeserializeOwned,
{
    let bytes: Vec<u8> = reader.read()?;
    bincode::deserialize(&bytes[..])
}

#[inline]
pub async fn async_serialize<W, T>(w: &mut IMuxAsync<W>, value: &T) -> Result<(), bincode::Error>
where
    W: futures::io::AsyncWrite + Unpin,
    T: serde::Serialize + ?Sized,
{
    let bytes: Vec<u8> = bincode::serialize(value)?;
    w.write(&bytes).await?;
    Ok(())
}

#[inline]
pub async fn async_deserialize<R, T>(reader: &mut IMuxAsync<R>) -> bincode::Result<T>
where
    R: futures::io::AsyncRead + Unpin,
    T: serde::de::DeserializeOwned,
{
    let bytes = reader.read().await?;
    bincode::deserialize(&bytes[..])
}
