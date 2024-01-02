from consts import *
from montecarlo import MonteCarlo
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class Bot():
    def __init__(self, api):
        self.api = api

    def close_all(self):
        self.api.close_all_positions()

    def make_buys(self):
        cash = float(self.api.get_account().cash)*0.9
        table = MonteCarlo(API).buys
        for i,r in table.iterrows():
            self.buy(r["stocks"], int(cash*r["weights"]/r["prices"]))


    
    def buy(self, code, qty):
        order_data = MarketOrderRequest(
            symbol = code,
            qty = qty,
            side = OrderSide.BUY,
            time_in_force = TimeInForce.DAY
        )
        API.submit_order(order_data=order_data)


b = Bot(API)
# b.make_buys()
b.close_all()
