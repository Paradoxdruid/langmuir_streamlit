import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# # Functions ##


# Define the model
def langmuir_model(x, Bmax, hill, kd):
    return Bmax * (x**hill) / (kd**hill + x**hill)


def langmuir_fitting(x, y):
    # Create a parameters object for the model
    params = lmfit.Parameters()
    params.add("Bmax", value=1.0, min=0)
    params.add("hill", value=1.0, min=0, max=8)
    params.add("kd", value=1.0, min=0)

    # Define the objective function to minimize (residual sum of squares)
    def objective(params, x, y):
        return (
            y - langmuir_model(x, params["Bmax"], params["hill"], params["kd"])
        ) ** 2

    # Perform the fit using the lmfit module
    result = lmfit.minimize(objective, params, args=(x, y))

    # Extract the values from the fitted parameters
    Bmax = result.params["Bmax"].value
    hill = result.params["hill"].value
    kd = result.params["kd"].value

    # Calculate the R-squared value of the fit
    y_pred = langmuir_model(x, Bmax, hill, kd)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.average(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # print(f"{Bmax=}")
    # print(f"{hill=}")
    # print(f"{kd=}")
    # print(f"{r_squared=}")
    return Bmax, hill, kd, r_squared


def two_site_langmuir_model(x, Bmax1, hill1, kd1, Bmax2, hill2, kd2):
    return Bmax1 * (x**hill1) / (kd1**hill1 + x**hill1) + Bmax2 * (x**hill2) / (
        kd2**hill2 + x**hill2
    )


def two_site_langmuir_fitting(x, y):
    # Create a parameters object for the model
    params = lmfit.Parameters()
    params.add("Bmax1", value=1.0, min=0)
    params.add("hill1", value=1.0, min=0, max=8)
    params.add("kd1", value=1.0, min=0)
    params.add("Bmax2", value=1.0, min=0)
    params.add("hill2", value=1.0, min=0, max=8)
    params.add("kd2", value=1.0, min=0)

    # Define the objective function to minimize (residual sum of squares)
    def objective(params, x, y):
        return (
            y
            - two_site_langmuir_model(
                x,
                params["Bmax1"],
                params["hill1"],
                params["kd1"],
                params["Bmax2"],
                params["hill2"],
                params["kd2"],
            )
        ) ** 2

    # Perform the fit using the lmfit module
    result = lmfit.minimize(objective, params, args=(x, y))

    # Extract the values from the fitted parameters
    Bmax1 = result.params["Bmax1"].value
    hill1 = result.params["hill1"].value
    kd1 = result.params["kd1"].value
    Bmax2 = result.params["Bmax2"].value
    hill2 = result.params["hill2"].value
    kd2 = result.params["kd2"].value

    # Calculate the R-squared value of the fit
    y_pred = two_site_langmuir_model(x, Bmax1, hill1, kd1, Bmax2, hill2, kd2)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.average(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # print(f"{Bmax1=}")
    # print(f"{hill1=}")
    # print(f"{kd1=}")
    # print(f"{Bmax2=}")
    # print(f"{hill2=}")
    # print(f"{kd2=}")
    # print(f"{r_squared=}")
    return Bmax1, hill1, kd1, Bmax2, hill2, kd2, r_squared


st.title("Langmuir Fitting")

st.markdown(
    """
Please upload an excel (.xlsx) file with Target [] in column A and Current response in column B, with a header row 1."""
)

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    x = df.iloc[1:, 0].values
    y = df.iloc[1:, 1].values

    Bmax, hill, kd, r_squared = langmuir_fitting(x, y)

    fake_x = np.linspace(0, max(x), 30000)
    fit_y = langmuir_model(fake_x, Bmax, hill, kd)

    with plt.style.context("seaborn-deep"):
        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(10)
        plt.rcParams.update({"font.size": 16})
        plt.plot(x, y, "o")
        plt.plot(fake_x, fit_y)
        plt.annotate(f"Kd = {kd:.2e} M", xy=(0.2, 0.5), xycoords="axes fraction")
        plt.annotate(f"hill = {hill:.2g}", xy=(0.2, 0.4), xycoords="axes fraction")
        plt.title(f"{uploaded_file.name}")
        plt.xlabel("[Target], M")
        plt.ylabel("Current, nA")
        plt.xscale("log")
        plt.figure(facecolor="white")

        st.pyplot(fig)
