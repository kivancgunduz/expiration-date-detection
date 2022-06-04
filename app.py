import uvicorn
from Utils.routes import Routes

routes = Routes()
app = routes.create()


if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.2")