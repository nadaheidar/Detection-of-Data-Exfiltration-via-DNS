from kafka import KafkaConsumer
import  pandas as pd
def consumer_data():

        consumer = KafkaConsumer(
                'ml-raw-dns',
                bootstrap_servers=['localhost:9092'],
                auto_offset_reset='earliest',
                enable_auto_commit=False,
            )
        count=0
        list_data = []
        for domain in consumer:
                list_data.append(domain.value.decode('utf8'))

                if count>=100000:
                        break
                count += 1

        data ={
                'domain':list_data
        }
        df = pd.DataFrame(data)
        return df
