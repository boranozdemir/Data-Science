import numpy as np
#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

import pandas as pd

prs = pd.read_csv("Borano Lecture Notes/Python_Programming_for_Datascience_Boran/datasets/persona.csv")
prs.describe()

def check_df(dataframe, head=5):
    print("############## Shape ##############")
    print(dataframe.shape)
    print("############## Types ##############")
    print(dataframe.dtypes)
    print("############## Head ##############")
    print(dataframe.head(head))
    print("############## Tail ##############")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(prs)

"""
############## Shape ##############
(5000, 5)
############## Types ##############
PRICE       int64
SOURCE     object
SEX        object
COUNTRY    object
AGE         int64
dtype: object
############## Head ##############
   PRICE   SOURCE   SEX COUNTRY  AGE
0     39  android  male     bra   17
1     39  android  male     bra   17
2     49  android  male     bra   17
3     29  android  male     tur   17
4     49  android  male     tur   17
############## Tail ##############
      PRICE   SOURCE     SEX COUNTRY  AGE
4995     29  android  female     bra   31
4996     29  android  female     bra   31
4997     29  android  female     bra   31
4998     39  android  female     bra   31
4999     29  android  female     bra   31
##################### NA #####################
PRICE      0
SOURCE     0
SEX        0
COUNTRY    0
AGE        0
dtype: int64
##################### Quantiles #####################
        count     mean        std   min    0%  ...   50%   95%   99%  100%   max
PRICE  5000.0  34.1320  12.464897   9.0   9.0  ...  39.0  49.0  59.0  59.0  59.0
AGE    5000.0  23.5814   8.995908  15.0  15.0  ...  21.0  43.0  53.0  66.0  66.0

"""


# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

prs["SOURCE"].nunique()
"2"

# Soru 3: Kaç unique PRICE vardır?

prs["PRICE"].nunique()
"6"

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

prs["PRICE"].value_counts()

"""
29    1305
39    1260
49    1031
19     992
59     212
9      200
"""

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

prs["COUNTRY"].value_counts()
"""
usa    2065
bra    1496
deu     455
tur     451
fra     303
can     230
"""

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

prs.groupby("COUNTRY").agg({"PRICE":"sum"})

"""
        PRICE
COUNTRY       
bra      51354
can       7730
deu      15485
fra      10177
tur      15689
usa      70225
"""


# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?

prs["SOURCE"].value_counts()
"""
android    2974
ios        2026
Name: SOURCE, dtype: int64
"""

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

prs.groupby("COUNTRY").agg({"PRICE":"mean"})

"""
            PRICE
COUNTRY           
bra      34.327540
can      33.608696
deu      34.032967
fra      33.587459
tur      34.787140
usa      34.007264

"""


# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

prs.groupby("SOURCE").agg({"PRICE":"mean"})

"""
            PRICE
SOURCE            
android  34.174849
ios      34.069102
"""

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

prs.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE":"mean"})

"""
                    PRICE
COUNTRY SOURCE            
bra     android  34.387029
        ios      34.222222
can     android  33.330709
        ios      33.951456
deu     android  33.869888
        ios      34.268817
fra     android  34.312500
        ios      32.776224
tur     android  36.229437
        ios      33.272727
usa     android  33.760357
        ios      34.371703
"""


#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################

prs.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE":"mean"})






#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

agg_df = prs.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE":"mean"}).sort_values("PRICE", ascending=False)




#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()
# agg_df.reset_index(inplace=True)

agg_df = agg_df.reset_index()




#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'

prs["AGE"] = prs["AGE"].astype("category")
agg_df["AGE_GROUPS"] = prs["AGE"]


age_space = [0, 18, 23, 30, 40, 70]

label = ['0_18', '19_23', '24_30', '31_40', '41_70']

agg_df["AGE_GROUPS"] = pd.cut(agg_df["AGE"], bins=age_space, labels=label)



#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.


agg_df["customers_level_based"] = np.nan
agg_df["customers_level_based"] = [col[0].upper() + '_' + col[1].upper() + '_' + col[2].upper() + "_" + col[5].upper() for col in agg_df.values]
agg_df.drop(["COUNTRY", "SOURCE", "SEX", "AGE", "AGE_GROUPS"], axis=1, inplace=True)
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})



#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.reset_index(inplace=True)

agg_df.groupby("SEGMENT").agg({"PRICE": ["min","max","mean","sum"]})



#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] ==new_user]

"""
72  TUR_ANDROID_FEMALE_31_40  41.833333       A
"""


# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?

new_user2 = "FRA_IOS_FEMALE_31_40"

agg_df[agg_df["customers_level_based"] ==new_user2]

"""
63  FRA_IOS_FEMALE_31_40  32.818182       C
"""