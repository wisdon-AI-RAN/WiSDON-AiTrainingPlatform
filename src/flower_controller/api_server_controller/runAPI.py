#==========================================================
# Description: Running API server/client for AI Training Platform.
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2026/01/20
# Version: 0.1.0
# License: None
#==========================================================

import uvicorn
from buildAPI import app

if __name__ == "__main__":
    print("🚀 Starting User Info API Server")
    print("📖 API Documentation: http://127.0.0.1:3032/docs")
    print("🔗 API Base URL: http://127.0.0.1:3032")
    print("=" * 50)
    
    uvicorn.run(
        "buildAPI:app",
        host="0.0.0.0",
        port=3032,
        reload=True,
        log_level="info"
    ) 