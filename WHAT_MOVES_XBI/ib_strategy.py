from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import numpy as np
import time

class MomentumStrategy(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self.positions = {}
        
    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId} {errorCode} {errorString}")
        
    def historicalData(self, reqId, bar):
        if reqId not in self.data:
            self.data[reqId] = [{"Date":bar.date,"Open":bar.open,"High":bar.high,"Low":bar.low,"Close":bar.close,"Volume":bar.volume}]
        else:
            self.data[reqId].append({"Date":bar.date,"Open":bar.open,"High":bar.high,"Low":bar.low,"Close":bar.close,"Volume":bar.volume})
        
    def historicalDataEnd(self, reqId, start, end):
        super().historicalDataEnd(reqId, start, end)
        print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)
        self.data[reqId] = pd.DataFrame(self.data[reqId])
        self.data[reqId].set_index("Date",inplace=True)
        self.momentum_strategy(reqId)
        
    def momentum_strategy(self, reqId):
        data = self.data[reqId]
        data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
        data["position"] = np.sign(data["returns"].rolling(window=60,min_periods=1).mean())
        
        if data["position"].iloc[-1] == 1:
            if reqId not in self.positions:
                self.placeSampleOrder(reqId)
                self.positions[reqId] = 1
        elif data["position"].iloc[-1] == -1: 
            if reqId in self.positions:
                self.cancelSampleOrder(reqId)
                self.positions.pop(reqId)
                
    def placeSampleOrder(self, reqId):
        contract = Contract()
        contract.symbol = self.ib_map[reqId]
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.primaryExchange = "NASDAQ"
        
        self.placeOrder(reqId, contract, self.sampleOrder(reqId))
        
    def cancelSampleOrder(self, reqId):
        self.cancelOrder(reqId)
        
    @staticmethod
    def sampleOrder(orderId):
        order = Order()
        order.orderId = orderId
        order.action = "BUY"
        order.totalQuantity = 100
        order.orderType = "MKT"
        return order

def main():
    app = MomentumStrategy()
    
    app.connect("127.0.0.1", 7497, clientId=1)
    
    time.sleep(1)
    
    tickers = ["AMGN", "GILD", "REGN", "VRTX"]
    app.ib_map = {}
    for i in range(len(tickers)):
        app.ib_map[i] = tickers[i]
        
    for i in range(len(tickers)):
        queryTime = (datetime.datetime.today() - datetime.timedelta(days=180)).strftime("%Y%m%d %H:%M:%S")
        app.reqHistoricalData(reqId=i, 
                              contract=Contract(),
                              endDateTime='',
                              durationStr="6 M",
                              barSizeSetting="1 day",
                              whatToShow="TRADES",
                              useRTH=1,
                              formatDate=1,
                              keepUpToDate=0,
                              chartOptions=[])
        time.sleep(10)
        
    app.run()
    
if __name__ == "__main__":
    main()