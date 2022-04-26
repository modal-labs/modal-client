async def asgi_app_wrapper(asgi_app, scope, body=None):
    messages = []

    async def send(message):
        print("SENT", message)
        messages.append(message)

    async def receive():
        print("RECEIVED", body)
        return {"type": "http.request", "body": body}

    print("INSIDE WRAPPER!", scope)
    await asgi_app(scope, receive, send)

    return messages
