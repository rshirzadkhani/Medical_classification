import pandas as pd

# order_items    = pd.read_csv("./data/olist_order_items_dataset.csv")
# orders         = pd.read_csv("./data/olist_orders_dataset.csv")
# order_payments = pd.read_csv("./data/olist_order_payments_dataset.csv")
# products       = pd.read_csv("./data/olist_products_dataset.csv")
# customers      = pd.read_csv("./data/olist_customers_dataset.csv")
# sellers        = pd.read_csv("./data/olist_sellers_dataset.csv")
# reviews        = pd.read_csv("./data/olist_order_reviews_dataset.csv")
# product_category_translation = pd.read_csv("./data/product_category_name_translation.csv")

# print(order_items.dtypes)
# print(orders.dtypes)
# print(order_payments.dtypes)
# print(products.dtypes)
# print(customers.dtypes)
# print(sellers.dtypes)
# print(reviews.dtypes)
# print(product_category_translation.dtypes)

# merged = order_items.merge(orders, on='order_id') \
#                     .merge(order_payments, on='order_id') \
#                     .merge(products, on='product_id') \
#                     .merge(customers, on='customer_id') \
#                     .merge(sellers, on='seller_id') \
#                     .merge(product_category_translation, on='product_category_name')

# merged.to_csv('./data/brazilian_ecommerce_dataset.csv', index=False)
data = pd.read_csv("./data/brazilian_ecommerce_dataset.csv")

data = data.dropna()
