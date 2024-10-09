fn main() -> anyhow::Result<()> {
    tonic_build::configure().bytes(["."]).compile_protos(
        &[
            "../../modal_proto/api.proto",
            "../../modal_proto/options.proto",
        ],
        &["../.."],
    )?;
    Ok(())
}
