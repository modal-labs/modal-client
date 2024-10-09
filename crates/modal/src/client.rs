use crate::auth;
use crate::config;
use crate::error;
use crate::function;
use crate::schema;
use crate::schema::modal_client_client;
use tonic::metadata;
use tonic::transport;

#[derive(Clone, Debug)]
pub struct Client {
    client: auth::AuthedClient,
}

impl Client {
    pub async fn connect(
        config: config::Config,
        workspace: Option<&str>,
    ) -> Result<Self, error::Error> {
        use std::str::FromStr as _;

        let client_id;
        let client_secret;

        if let (Some(id), Some(secret)) = (config.task_id, config.task_secret) {
            let id = metadata::MetadataValue::from_str(&id)?;
            let mut secret = metadata::MetadataValue::from_str(&secret)?;
            secret.set_sensitive(true);

            client_id = Some(("x-modal-task-id", id));
            client_secret = Some(("x-modal-task-secret", secret));
        } else if let (Some(id), Some(secret)) = (config.token_id, config.token_secret) {
            let id = metadata::MetadataValue::from_str(&id)?;
            let mut secret = metadata::MetadataValue::from_str(&secret)?;
            secret.set_sensitive(true);

            client_id = Some(("x-modal-token-id", id));
            client_secret = Some(("x-modal-token-secret", secret));
        } else {
            client_id = None;
            client_secret = None;
        }

        let workspace = if let Some(w) = workspace {
            Some((
                "x-modal-workspace",
                metadata::MetadataValue::from_str(w)?,
            ))
        } else {
            None
        };

        let server_url = config
            .server_url
            .unwrap_or_else(|| "https://api.modal.com".to_owned());
        let channel = transport::Channel::from_shared(server_url)?
            .tls_config(transport::ClientTlsConfig::new().with_native_roots())?
            .connect()
            .await?;
        let auth_interceptor = auth::AuthInterceptor::new(client_id, client_secret, workspace);
        let client =
            modal_client_client::ModalClientClient::with_interceptor(channel, auth_interceptor);

        Ok(Self { client })
    }

    pub async fn lookup_function(
        &mut self,
        environment: &str,
        app: &str,
        name: &str,
    ) -> Result<function::Function, error::Error> {
        let fn_get = self
            .client
            .function_get(schema::FunctionGetRequest {
                app_name: app.to_owned(),
                object_tag: name.to_owned(),
                namespace: schema::DeploymentNamespace::Workspace as i32,
                environment_name: environment.to_owned(),
            })
            .await?
            .into_inner();

        let metadata = fn_get.handle_metadata.unwrap_or_default();
        // TODO: support other kinds of functions
        assert!(!metadata.is_method);
        assert_eq!(
            schema::function::FunctionType::Function as i32,
            metadata.function_type
        );

        let client = self.client.clone();
        let id = fn_get.function_id;
        Ok(function::Function::new(id, client))
    }
}
