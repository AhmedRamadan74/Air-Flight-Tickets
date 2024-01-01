import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
import plotly.express as px
import xgboost
import joblib
import math
import datetime

# layout
st.set_page_config(page_title="Air Flight Tickets", layout="wide")
origanal_data = pd.read_excel("Data_Train.xlsx")
df = pd.read_csv("clean_data.csv")
df_min_max_duration = pd.read_csv("min_max_duration.csv")


# Visualization Function
def bar(data_frame, x, y, title_text, color=None):
    fig = px.bar(
        data_frame=data_frame, x=x, y=y, color=color, barmode="group", text_auto="0.2s"
    )
    fig.update_traces(textfont_size=12, textposition="outside")
    fig.update_layout(title_text=title_text, title_x=0.5)
    return fig


# Visualization Function
def bar(data_frame, x, y, title_text, color=None):
    fig = px.bar(
        data_frame=data_frame, x=x, y=y, color=color, barmode="group", text_auto="0.2s"
    )
    fig.update_traces(textfont_size=12, textposition="outside")
    fig.update_layout(title_text=title_text, title_x=0.5)
    return fig


def sunburst(data, names, path, values, title_text, width=700, height=600):
    fig = px.sunburst(
        data_frame=data,
        names=names,
        path=path,
        values=values,
        width=width,
        height=height,
    )
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(title_text=title_text, title_x=0.5)
    return fig


pages = st.sidebar.radio("Pages", ["Data", "Analysis", "Predict price your ticket"])
######################################################################################################################################
if pages == "Data":
    st.subheader("Data Describtion : ")
    st.write(
        "This data was collected in 2019 and its about detials of air flight tickets in india and in this project will predict price of tickets."
    )
    st.write(
        """
             | Attribute | Description |
             |----------|----------|
             |Airline|an airline that booked ticket|
             |Date_of_Journey| the date of journey|
             |Source|The place of take-off|
             |Destination|The landing place|
             |Route| path of journey|
             |Dep_Time|represent time of take-off|
             |Arrival_Time| a time when arrived flight|
             |Duration|duration of journey ( difference between Arrival_Time and  Dep_Time)|
             |Total_Stops|represent if flight fly from Source to Destination with rest or not|
             |Price|price of air flight tickets|
             """
    )
    st.subheader("Display first 10 rows of data: ")
    st.dataframe(origanal_data.head(10))
######################################################################################################################################
if pages == "Analysis":
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
        (0.1, 2.3, 0.1, 1.3, 0.1)
    )

    with row0_1:
        st.title("Analysis data")
        st.markdown(
            """ <h6>
                        You will be book flight ticket from city to city in india ,some analysis before book. </center> </h6> """,
            unsafe_allow_html=True,
        )
    with row0_2:
        st.text("")
        st.subheader(
            "Linkedin : App by [Ahmed Ramadan](https://www.linkedin.com/in/ahmed-ramadan-18b873230/) "
        )
        st.subheader(
            "Github : App by [Ahmed Ramadan](https://github.com/AhmedRamadan74/Air-Flight-Tickets)"
        )
    # Dividing our analysis into tabs, each tab contains
    over_view, Airline_companies, Conclusion = st.tabs(
        ["Over View", "Airline companies", "Conclusion"]
    )
    with over_view:
        st.title("Analytical overview :")
        # insights in this tab
        st.write("The information in this tab can answer the following questions :")
        st.write(" 1- What is the most popular airline company ?")
        st.write(" 2- What is the busier airport ?")
        st.write(" 3- What is the airline company have the highest average price ?")
        st.write(" 4- What is average Price for each  Total Stops ?")
        st.write(
            " 5- Average Price in Additional_Info and what is the type of Additional_Info have the highest and lowest average price ?"
        )
        st.write(
            "6- Average Duration_minute for each Source cities to Destination cities with Total_Stops ?"
        )

        st.write("*" * 50)
        # Q1
        st.subheader(" 1- What is the most popular airline company ?")
        # pandas
        data = df.Airline.value_counts().reset_index()
        fig = bar(
            data, "Airline", "count", "What is the most popular airline company ?"
        )
        st.plotly_chart(fig)
        st.write("- Jet Airways is the most popular airline to fly")
        st.markdown("*" * 50)
        # Q2
        st.subheader(" 2- What is the busier airport ?")
        # panads
        data_source = df.Source.value_counts().reset_index()
        data_dens = df.Destination.value_counts().reset_index()
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
            (0.1, 2.3, 0.1, 1.3, 0.1)
        )
        with row0_1:
            fig = bar(
                data_source,
                "Source",
                "count",
                "2- What is the busier airport in Source cities ?",
            )
            st.plotly_chart(fig)
        with row0_2:
            fig = bar(
                data_dens,
                "Destination",
                "count",
                "2- What is the busier airport in Destination cities ?",
            )
            st.plotly_chart(fig)
        st.write("- Delhi and Cochin are the busier airport")
        st.markdown("*" * 50)
        # Q3
        st.subheader(" 3- What is the airline company have the highest average price ?")
        # pandas
        data = (
            df.groupby("Airline")["Price"]
            .mean()
            .reset_index()
            .sort_values(by="Price", ascending=False)
        )
        fig = bar(
            data,
            "Airline",
            "Price",
            title_text="3- What is the airline company have the highest average price ?",
        )
        st.plotly_chart(fig)
        st.write("- Jet Airways	airline company that has  the highest average price ")
        st.markdown("*" * 50)
        # Q4
        st.subheader("4- What is average Price for each  Total Stops ?")
        # pandas
        data = (
            df.groupby("Total_Stops")["Price"]
            .mean()
            .reset_index()
            .sort_values(by="Price", ascending=False)
        )
        fig = bar(
            data,
            "Total_Stops",
            "Price",
            title_text=" What is average Price for each  Total Stops ?",
        )
        st.plotly_chart(fig)
        st.write("- The ticket will be more expensive the more Total Stops")
        st.markdown("*" * 50)
        # ََQ5
        st.subheader(
            " 5- Average Price in Additional_Info and what is the type of Additional_Info have the highest and lowest average price ?"
        )
        # pandas
        data = (
            df.groupby("Additional_Info")["Price"]
            .mean()
            .reset_index()
            .sort_values(by="Price", ascending=False)
        )
        fig = bar(
            data,
            "Additional_Info",
            "Price",
            title_text="Average Price in Additional_Info ?",
        )
        st.plotly_chart(fig)
        st.write(
            "- Business class has the highest average price and No check-in baggage included has lowest average price"
        )
        st.markdown("*" * 50)
        # Q6
        st.subheader(
            "6- Average Duration_minute for each Source cities to Destination cities with Total_Stops ?"
        )
        data = (
            df.groupby(["Source", "Destination", "Total_Stops"])
            .agg({"Duration_minute": "mean"})
            .reset_index()
            .sort_values(by="Duration_minute", ascending=False)
        )
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
            (0.1, 2.3, 0.1, 1.3, 0.1)
        )
        with row0_1:
            fig = sunburst(
                data,
                "Source",
                ["Source", "Destination", "Total_Stops"],
                "Duration_minute",
                " Average Duration_minute for each Source cities to Destination cities with Total_Stops? ",
                width=700,
                height=700,
            )
            st.plotly_chart(fig)
        with row0_2:
            st.dataframe(data)
        st.write("___Note:___")
        st.write("- 1) The higher the number of Total_Stops, the longer the trip")
        st.write("- 2) The longest time for a trip between Banglore and New Delhi")
        st.write("- 3) The shortest time for a trip between Mumbai and Hyderabad	")
        st.markdown("*" * 50)

    ################################################################################
    with Airline_companies:
        st.write(
            "The information in this tab can answer the following questions with specific airline company :"
        )
        st.write(
            " 1-  Average Price for each airline company for each Source cities to Destination cities ?"
        )
        st.write(
            " 2-  Average Price for each airline company for each Source cities to Destination cities with its Additional_Info ?"
        )
        st.write("*" * 50)
        # Q1
        st.subheader(
            "1-  Average Price for each airline company for each Source cities to Destination cities ?"
        )
        # pandas
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
            (0.1, 2.3, 0.1, 5, 10)
        )

        data = (
            df.groupby(["Airline", "Source", "Destination"])
            .agg({"Price": "mean"})
            .rename(columns={"Price": "Price"})
            .reset_index()
            .sort_values(by="Price", ascending=False)
        )
        with row0_1:
            fig = sunburst(
                data,
                "Airline",
                ["Airline", "Source", "Destination"],
                "Price",
                "Average Price for each airline company for each Source cities to Destination cities ? ",
                width=1000,
                height=1000,
            )
            st.plotly_chart(fig)
        with row0_spacer3:
            st.dataframe(data)

        st.write("__Note :__")
        st.write(
            "-  1) The expensive price ticket  from Banglore to New Delhi	with Jet Airways airline"
        )
        st.write(
            "-  2) The cheapest price ticket  from Mumbai to Hyderabad	with SpiceJet airline"
        )
        # Specific airline
        st.subheader(
            "Average Price for Specific airline company for each Source cities to Destination cities ?"
        )
        airline = st.selectbox(
            "Select Specific airline : ", data.Airline.unique().tolist()
        )
        data_Specific = data[data["Airline"] == airline]
        fig = sunburst(
            data_Specific,
            "Airline",
            ["Airline", "Source", "Destination"],
            "Price",
            f"Average Price for {airline} airline company for each Source cities to Destination cities ? ",
        )
        st.plotly_chart(fig)
        st.write("*" * 50)
        # Q2
        st.subheader(
            "2-  Average Price for each airline company for each Source cities to Destination cities with its Additional_Info ?"
        )
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
            (0.1, 2.3, 0.1, 5, 10)
        )
        data = (
            df.groupby(["Airline", "Source", "Destination", "Additional_Info"])
            .agg({"Price": "mean"})
            .rename(columns={"Price": "Price"})
            .reset_index()
            .sort_values(by="Price", ascending=False)
        )
        with row0_1:
            fig = sunburst(
                data,
                "Airline",
                ["Airline", "Source", "Destination", "Additional_Info"],
                "Price",
                " Average Price for each airline company for each Source cities to Destination cities with its Additional_Info ? ",
                width=1000,
                height=1000,
            )
            st.plotly_chart(fig)
        with row0_spacer3:
            st.dataframe(data)
            # Specific airline
            st.subheader(
                "Average Price for Specific airline company for each Source cities to Destination cities with its Additional_Info ?"
            )
        airline2 = st.selectbox(
            "Select Specific airline : ", data.Airline.unique().tolist()
        )
        data_Specific = data[data["Airline"] == airline2]
        fig = sunburst(
            data_Specific,
            "Airline",
            ["Airline", "Source", "Destination", "Additional_Info"],
            "Price",
            f"Average Price for Specific {airline2} company for each Source cities to Destination cities with its Additional_Info ? ",
        )
        st.plotly_chart(fig)
        st.write("*" * 50)
    ################################################################################
    with Conclusion:
        st.subheader("The insight that is extracted from analysis :")
        st.write("1) __Jet Airways__")
        st.write(">  - is the most popular airline to fly")
        st.write(">  - has the highest average price")
        st.write(
            ">  - The expensive price ticket from Banglore to New Delhi with Jet Airways airline"
        )
        st.write(" 2) __Delhi and Cochin are the busier airport__")
        st.write(" 3) __The ticket will be more expensive the more Total Stops__")
        st.write(
            " 4) __Business class has the highest average price and No check-in baggage included has lowest average price__"
        )
        st.write(" 5) __The higher the number of Total_Stops, the longer the trip__")
        st.write(" 6) __The longest time for a trip between Banglore and New Delhi__")
        st.write(" 7) __The shortest time for a trip between Mumbai and Hyderabad__")
        st.write(
            " 8) __The cheapest price ticket from Mumbai to Hyderabad with SpiceJet airline__"
        )
######################################################################################################################################
if pages == "Predict price your ticket":
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
        (0.1, 2.3, 0.1, 1.3, 0.1)
    )

    with row0_1:
        st.title("Prediction ticket Price")
        st.markdown(
            """ <h6>
                        You will be book flight ticket from city to city in india , predict your ticket price. </center> </h6> """,
            unsafe_allow_html=True,
        )
    with row0_2:
        st.text("")
        st.subheader(
            "Linkedin : App by [Ahmed Ramadan](https://www.linkedin.com/in/ahmed-ramadan-18b873230/) "
        )
        st.subheader(
            "Github : App by [Ahmed Ramadan](https://github.com/AhmedRamadan74/Air-Flight-Tickets)"
        )

    model = joblib.load("model.pkl")  # load model
    inputs = joblib.load("input.pkl")  # load input

    def Make_Prediction(
        Airline,
        Source,
        Destination,
        Total_Stops,
        Additional_Info,
        month_of_Journey,
        day_of_Journey,
        Duration_minute,
    ):
        data = pd.DataFrame(columns=inputs)
        data.at[0, "Airline"] = Airline
        data.at[0, "Source"] = Source
        data.at[0, "Destination"] = Destination
        data.at[0, "Total_Stops"] = Total_Stops
        data.at[0, "Additional_Info"] = Additional_Info
        data.at[0, "month_of_Journey"] = month_of_Journey
        data.at[0, "day_of_Journey"] = day_of_Journey
        data.at[0, "Duration_minute"] = Duration_minute

        # prediction output
        result = model.predict(data)
        return round(result[0], 2)

    st.write("Frist , Entry details  your Air Flight ")

    Airline = st.selectbox(
        " An airline that booked a ticket :", df.Airline.unique().tolist()
    )
    Source = st.selectbox(
        "The place of take-off :", df_min_max_duration.Source.unique().tolist()
    )
    Destination = st.selectbox(
        "The landing place :",
        df_min_max_duration[df_min_max_duration["Source"] == Source]["Destination"]
        .unique()
        .tolist(),
    )

    Date_of_Journey = st.date_input("The date of journey")
    Dep_Time = st.time_input("Time of take-off ")

    st.write(f"represent if flight fly from Source to Destination with rest or not :")
    value_total_stop = (
        df_min_max_duration[
            (df_min_max_duration["Source"] == Source)
            & (df_min_max_duration["Destination"] == Destination)
        ]["Total_Stops"]
        .unique()
        .tolist()
    )
    Total_Stops = st.selectbox("0 represented to no rest : ", value_total_stop)
    Total_Stops = int(Total_Stops)
    Additional_Info = st.selectbox(
        "some additional info required :", df.Additional_Info.unique().tolist()
    )

    Min_Duration_minute = df_min_max_duration[
        (df_min_max_duration["Source"] == Source)
        & (df_min_max_duration["Destination"] == Destination)
        & (df_min_max_duration["Total_Stops"] == Total_Stops)
    ]["Min_Duration_minute"].values[0]
    Min_Duration_minute = math.floor(Min_Duration_minute / 60)
    Max_Duration_minute = df_min_max_duration[
        (df_min_max_duration["Source"] == Source)
        & (df_min_max_duration["Destination"] == Destination)
        & (df_min_max_duration["Total_Stops"] == Total_Stops)
    ]["Max_Duration_minute"].values[0]
    Max_Duration_minute = math.floor(Max_Duration_minute / 60)

    st.write(
        f"Range time of arrival is between {Min_Duration_minute} hours  and {Max_Duration_minute} hours"
    )
    decimal_hours = st.number_input(
        label="Time of arrival - format (hour.minute)",
        min_value=float(Min_Duration_minute),
        max_value=float(Max_Duration_minute),
    )
    hours, minutes = divmod(
        int(decimal_hours * 60), 60
    )  # to convert decimal number to hour and minutes
    st.write(f"Time of arrival is {hours} hours and {minutes} minutes")

    Duration_minute = int(decimal_hours * 60)

    result = Make_Prediction(
        Airline,
        Source,
        Destination,
        Total_Stops,
        Additional_Info,
        Date_of_Journey.month,
        Date_of_Journey.day,
        Duration_minute,
    )
    btn = st.button("Predict")
    if btn:
        st.write(f"Price of ticket = ", result)
