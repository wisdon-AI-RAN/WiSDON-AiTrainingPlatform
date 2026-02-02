import sys
import time
import grpc
import pymongo
import logging
import configparser
import gRPCSymptomInfoQuery_Service.sir_proto.symptomInfo_pb2 as symptomInfo_pb2
import gRPCSymptomInfoQuery_Service.sir_proto.symptomInfo_pb2_grpc as symptomInfo_pb2_grpc

# Create logger
logger = logging.getLogger("symptomInfo_client.py")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] symptomInfo_client.py - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

class DBConnection():
    """
        This class facilitates the connection to a predefined MongoDB instance
        using host and port configurations from the 'config' module.

        Usage:
            1. Create an instance of DBConnection.
            2. Use db_connect_collection() to get a reference to a specific DataBase and collection.

        Example:
            db = DB_Connection('your_database_name', 'your_collection_name')
            my_collection = db.db_connect_collection()
    """
    def __init__(self, database : str = 'lvsd', collection : str = 'train'):
        self.database = database
        self.collection = collection
        
    # Connect to MongoDB without specific database and collection
    def db_client(self):
        client = pymongo.MongoClient('AutoFLSystem_mongo', 27017)
        return client
    
    # Connect to specific database and collection
    def db_connect_collection(self):
        db = self.db_client()
        db = getattr(db, self.database)
        db = db[self.collection]
        return db

class gRPCSymptomInfoQueryServicer():
    """
        A class for sending client information and receiving responses from the symptom information gRPC server.

        Attributes:
            secure (bool): A flag indicating whether to use secure communication with SSL.

        Methods:
            set_grpc_stub: Sets up gRPC stub for communicating with the gRPC server.
            register_and_getSymptom: Register to server and retrive the current symptom which need to retrain.

        Example:
            servicer = gRPCSymptomInfoResponseServicer(secure=True)
            servicer.register_and_getSymptom()
    """
    def __init__(self, secure : bool = False):
        self.secure = secure
        
        # Load client config
        self.config = load_config('/app/config.ini')
        
        self.clientId = self.config['General']['clientID']

    def set_grpc_stub(self):
        if self.secure:
            with open("./certificates/ca.crt", "rb") as f:
                certificate_chain = f.read()
            credentials = grpc.ssl_channel_credentials(root_certificates=certificate_chain)
            channel = grpc.secure_channel('{}:{}'.format(self.config['gRPCSymptomInfo']['ip'], self.config['gRPCSymptomInfo']['port']), credentials)
        else:
            channel = grpc.insecure_channel('{}:{}'.format(self.config['gRPCSymptomInfo']['ip'], self.config['gRPCSymptomInfo']['port']))
        stub = symptomInfo_pb2_grpc.SymptomInfoResponseServiceStub(channel)
        return stub

    def register_and_getSymptom(self):
        while True:
            try:
                # Create gRPC client stub
                stub = self.set_grpc_stub()
                
                # Login server and wait for fl retrain symptom check
                response_symptom = stub.RegisterAndGetSymptom(symptomInfo_pb2.RegisterId(id=self.clientId))
                
                # RL or Pretrained
                if response_symptom.symptom == 'RL':
                    return 'RL', 'Accepted'
                
                if response_symptom.symptom == 'pretrained':
                    return 'pretrained', 'Accepted'
                
                # Connect to db to get symptom volume
                db = DBConnection(database=response_symptom.symptom, collection='train').db_connect_collection()     
                
                # Check data volume whether fulfill the threshold
                client_data = db.count_documents({})
                if client_data >= 100:
                    permission = 'Accepted'
                else:
                    permission = 'Denied'
                
                logger.info(f"Symptom Retrain Infomation {response_symptom.symptom} : {permission}")
                
                # Close gRPC channel
                stub = None
                
                return response_symptom.symptom, permission
            except grpc.RpcError as e:
                logger.error(f"Fail to connect to gRPC server !")
                time.sleep(5)