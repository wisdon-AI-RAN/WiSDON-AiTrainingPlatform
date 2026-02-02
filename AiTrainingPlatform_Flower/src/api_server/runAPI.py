#==========================================================
# Description: Running API server for Flower Server.
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2025/12/24
# Version: 0.1.0
# License: None
#==========================================================

import uvicorn
from buildAPI import app

if __name__ == "__main__":
    uvicorn.run(
        "buildAPI:app",
        host="0.0.0.0",
        port=3032,
        reload=True,
        log_level="info"
    ) 