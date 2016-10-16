from ib.ext.Contract import Contract
from ib.ext.Order import Order
from ib.opt import Connection, message

#def errorHandler(msg=''):
#    """Handles the capturing of error messages"""
#    print("Server Error: %s", msg)

class Trader(object):

    def __init__(self):
        self.orderId = 1

    def sendTradeInstruction(self, symbol, action, lots):
        """Sends an order to IB"""
    
        conn = self.createConnection()

        contract = self.createContract(symbol, 'STK', 'SMART', 'SMART', 'USD')
        order = self.createOrder('MKT', lots, 'BUY') 

        conn.placeOrder(self.orderId, contract, order)
        conn.disconnect()

        self.orderId += 1

    def errorHandler(self, msg=''):
        """Handles the capturing of error messages"""
        print("Server Error: %s", msg)

    def replyHandler(self, msg):
        """Handles of server replies"""
        print("Server Response: %s, %s", msg.typeName, msg)

    def createContract(self, symbol, sec_type, exch, prim_exch, curr):
        """Create a Contract object defining what will
        be purchased, at which exchange and in which currency.

        symbol - The ticker symbol for the contract
        sec_type - The security type for the contract ('STK' is 'stock')
        exch - The exchange to carry out the contract on
        prim_exch - The primary exchange to carry out the contract on
        curr - The currency in which to purchase the contract"""
        contract = Contract()
        contract.m_symbol = symbol
        contract.m_secType = sec_type
        contract.m_exchange = exch
        contract.m_primaryExch = prim_exch
        contract.m_currency = curr
        return 

    def createOrder(self, order_type, quantity, action):
        """Create an Order object (Market/Limit) to go long/short.

        order_type - 'MKT', 'LMT' for Market or Limit orders
        quantity - Integral number of assets to order
        action - 'BUY' or 'SELL'"""
        order = Order()
        order.m_orderType = order_type
        order.m_totalQuantity = quantity
        order.m_action = action
        return order

    def createConnection(self):
        conn = Connection.create(port=7497, clientId=102)
        
        conn.connect()

        # Assign the error handling function defined above
        # to the TWS connection
        conn.register(self.errorHandler, 'Error')

        # Assign all of the server reply messages to the
        # reply_handler function defined above
        conn.registerAll(self.replyHandler)

        return conn