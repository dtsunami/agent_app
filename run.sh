# app port/workers
setenv APP_PORT 3434
setenv APP_WORKERS 1

# run server
gunicorn server:app --timeout 0 --workers $APP_WORKERS --worker-class uvicorn.workers.UvicornWorker --bind ${HOSTNAME}:${APP_PORT} 
