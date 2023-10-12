from datetime import datetime
import numpy as np
import pandas as pd


def webapp_data_processing(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function processes a dataset composed of the metadata plus the plumes into a new dataset with wanted KPIs.

    Args:
    :param data: Input pandas dataframe

    Returns:
    :returns: pandas dataframe with KPIs of interest
    """
    data['date'] =  data['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
    data['count'] = data.groupby(['lat', "lon"]).transform('size')
    data['last_date'] = data.groupby(["lat", "lon"])['date'].transform('max')
    data["plume"] = np.where(data["plume"] == "yes",1, 0)
    data['plume_at_last_date'] = np.where(data['date'] == data['last_date'], np.where(data["plume"] == 1,1,0), 0)
    data["plume_at_last_date"] = data.groupby(["lat", "lon"])['plume_at_last_date'].transform('max')
    data['threshold_date'] = (data["date"] >= (data['last_date'] - pd.DateOffset(months=1)))
    data["plume_last_month"] =  np.where(((data['threshold_date']) & (data["plume"] == 1) ), True, False)
    data["last_month"] =  np.where((data['threshold_date']), True, False)
    data = (data.groupby(['lat','lon','count',"last_date", "plume_at_last_date"])
         .apply(lambda x: ((x['plume_last_month']).sum(), (x["last_month"]).sum(), (x["plume"]).sum()))
         .reset_index(name='New_count'))
    data['plume_count_lm'] = data["New_count"].apply(lambda x: x[0])
    data["total_count_lm"] = data["New_count"].apply(lambda x: x[1])
    data["plume_count"] = data["New_count"].apply(lambda x: x[2])
    data["Responsible"] = "joaohfpmelo@gmail.com"
    data = data.drop("New_count", axis=1)
    return data


def mapsize(selected_metadata: pd.DataFrame)-> tuple[list, list] :
    """
    This function takes a dataset and returns the min and max coordinates.
    Args:
    :param selected_metadata: Input pandas dataframe

    Returns:
    :returns: Tuple consisting of min and max coordinates
    """
    coords = []
    for i in range(0, len(selected_metadata)):
        coords += [(selected_metadata.iloc[i]["lat"], selected_metadata.iloc[i]["lon"])]
    south_west_corner = min(coords)
    north_east_corner = max(coords)
    return (south_west_corner, north_east_corner)



def table(left_col_color: str, right_col_color: str, selected_metadata:pd.DataFrame, i:int) -> str:
    """
    This function takes two colors, a dataset and an int and returns an html table.
    Args:
    :param left_col_color: Color of left side of table
    :param right_col_color: Color of right side of table
    :param selected_metadata: Input pandas dataframe
    :param i: Int which selects the row of the dataframe for which the table is built

    Returns:
    :returns: Html code for a table with information on KPI's
    """
    return  f"""
            <center> <table style="height: 126px; width: 305px;">
            <tbody>
            <tr>
            <td style="width: 250px;background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Date of last picture </span></td>
            <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["last_date"]) + """
            </tr>
            <tr>
            <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Last known leakage state </span></td>
            <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["plume_at_last_date"]) + """
            </tr>
            <tr>
            <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Total number of pictures </span></td>
            <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["count"]) + """
            </tr>
            <tr>
            <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Total number of leakages </span></td>
            <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["plume_count"]) + """
            </tr>
            <tr>
            <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Number of pictures last month </span></td>
            <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["total_count_lm"]) + """
            </tr>
            <tr>
            <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Number of leakages last month </span></td>
            <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["plume_count_lm"]) + """
            </tr>
            <tr>
            <td style="background-color: """+ left_col_color +""";"><span style="color: #ffffff;">Responsible </span></td>
            <td style="width: 250px;background-color: """+ right_col_color +""";">{}</td>""".format(selected_metadata.iloc[i]["Responsible"]) + """
            </tr>
            </tbody>
            </table></center>
                """

