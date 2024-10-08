use crate::schema::modal_client_client;
use tonic::metadata;
use tonic::service::interceptor;
use tonic::transport;

pub type AuthedClient = modal_client_client::ModalClientClient<
    interceptor::InterceptedService<transport::Channel, AuthInterceptor>,
>;

pub type AuthField = Option<(&'static str, metadata::MetadataValue<metadata::Ascii>)>;

#[derive(Clone, Debug)]
pub struct AuthInterceptor {
    client_id: AuthField,
    client_secret: AuthField,
    workspace: AuthField,
}

impl AuthInterceptor {
    pub fn new(client_id: AuthField, client_secret: AuthField, workspace: AuthField) -> Self {
        Self {
            client_id,
            client_secret,
            workspace,
        }
    }
}

impl interceptor::Interceptor for AuthInterceptor {
    fn call(&mut self, mut req: tonic::Request<()>) -> Result<tonic::Request<()>, tonic::Status> {
        if let Some((field, value)) = self.client_id.as_ref() {
            req.metadata_mut().insert(*field, value.to_owned());
        }
        if let Some((field, value)) = self.client_secret.as_ref() {
            req.metadata_mut().insert(*field, value.to_owned());
        }
        if let Some((field, value)) = self.workspace.as_ref() {
            req.metadata_mut().insert(*field, value.to_owned());
        }
        Ok(req)
    }
}
