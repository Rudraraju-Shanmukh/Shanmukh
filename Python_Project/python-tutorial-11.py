##### Python Modules ######
# Writing Python modules

# 1. import module_name
import pricing
# module_name.function_name()

net_price = pricing.get_net_price(
    price=100,
    tax_rate=0.01
)

print(net_price)

# 2. import <module_name> as new_name
import pricing as selling_price

net_price = selling_price.get_net_price(
    price=100,
    tax_rate=0.01
)

###### Python Module Search Path ######

import sys

for path in sys.path:
    print(path)

sys.path.append('d:\\modules\\')

# main.py
import pricing


net_price = pricing.get_net_price(
    price=100,
    tax_rate=0.01
)

print(net_price)

# 3. from <module_name> import <name>

from pricing import get_net_price

net_price = get_net_price(price=100, tax_rate=0.01)
print(net_price)

# 4. from <module_name> import <name> as <new_name>: rename the imported objects

from pricing import get_net_price as calculate_net_price

net_price = calculate_net_price(
    price=100,
    tax_rate=0.1,
    discount=0.05
)

# 5.  from <module_name> import * : import all objects from a module

from pricing import *
from product import *

tax = get_tax(100)
print(tax)

from sales.order import create_sales_order
from sales.delivery import create_delivery
from sales.billing import create_billing


create_sales_order()
create_delivery()
create_billing()

