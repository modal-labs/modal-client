use modal::args;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = modal::Config::load()?;
    let mut client = modal::Client::connect(config, None).await?;

    let mut func = client.lookup_function("main", "payload-value", "f").await?;
    let result = func.call(args!(3, bar = "hi")).await?;
    eprintln!("result = {:?}", result);
    Ok(())
}
