#[inline]
pub fn serialize<W: std::io::Write, T: ?Sized>(mut w: W, value: &T) -> Result<(), bincode::Error>
where
    T: serde::Serialize,
{
    let bytes: Vec<u8> = bincode::serialize(value)?;
    let _ = w.write(&bytes)?;
    Ok(())
}

#[inline]
pub fn deserialize<R, T>(reader: R) -> bincode::Result<T>
where
    R: std::io::Read,
    T: serde::de::DeserializeOwned,
{
    bincode::deserialize_from(reader)
}

// TODO
use async_std::prelude::*;
use std::convert::TryFrom;

#[inline]
pub async fn async_serialize<W, T>(mut w: W, value: &T) -> Result<(), bincode::Error>
where
    W: async_std::io::Write + Unpin,
    T: serde::Serialize + ?Sized,
{
    let bytes: Vec<u8> = bincode::serialize(value)?;
    w.write_all(&(bytes.len() as u64).to_le_bytes()).await?;
    w.write_all(&bytes).await?;
    Ok(())
}

#[inline]
pub async fn async_deserialize<R, T>(mut reader: R) -> bincode::Result<T>
where
    R: async_std::io::Read + Unpin,
    T: serde::de::DeserializeOwned,
{
    // Read the message length
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf).await?;
    let len: u64 = u64::from_le_bytes(len_buf);
    // Read the rest of the message
    let mut buf = vec![0u8; usize::try_from(len).unwrap()];
    reader.read_exact(&mut buf[..]).await?;
    bincode::deserialize_from(&buf[..])
}
