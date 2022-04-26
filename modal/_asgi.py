async def asgi_app_wrapper(asgi_app, scope, body=None):
    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        return {"type": "http.request", "body": body}

    await asgi_app(scope, receive, send)

    return messages
