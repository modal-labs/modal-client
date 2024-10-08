use modal::args;
use modal::kwargs;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = modal::Config::load()?;
    let mut client = modal::Client::connect(config, Some("dflemstr")).await?;

    let mut func = client
        .lookup_function("main", "example-lifecycle-web", "hello")
        .await?;
    func.call(args!["hi", 1, true], kwargs! {foo=3, bar=4}).await?;
    Ok(())
}
