import pandas as pd

def data_loader():
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
    data = data[data['customer_state'].isin(["SP","RJ","MG"])]
    data = data[['product_category_name','product_photos_qty',
                'product_weight_g', 'product_length_cm',
                'product_height_cm','product_width_cm',
                'customer_state','price']]

    data = data.dropna()
    print(data.dtypes)
    # Remove outliers
    data = data[(data["price"] >= data["price"].quantile(0.05)) & 
                (data["price"] <= data["price"].quantile(0.95))]

    # Convert categorical variables to numerical values
    data = pd.get_dummies(data, columns=['product_category_name', 'customer_state'])

    return data

if __name__ == "__main__":
    data_loader()