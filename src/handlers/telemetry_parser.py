import scipy.io
import numpy as np

def filter_data(mat_file):
    """
    loads a .mat telemetry file, performs interpolation to a common time axis
    and trims data based on gps signal accuracy.
    """
    print(f"Loading and filtering data from file: {mat_file}")
    mat_data = scipy.io.loadmat(mat_file)
    
    # gps and speed (ins)
    longitude = np.array(mat_data['INS'][0,0]['EKF_POS'][0,0]['Longitude']).flatten()
    latitude = np.array(mat_data['INS'][0,0]['EKF_POS'][0,0]['Latitude']).flatten()
    gps_time = np.array(mat_data['INS'][0,0]['EKF_POS'][0,0]['t']).flatten()

    altitude = np.array(mat_data['INS'][0,0]['EKF_ALTITUDE'][0,0]['Altitude']).flatten()
    altitude_time =  np.array(mat_data['INS'][0,0]['EKF_ALTITUDE'][0,0]['t']).flatten()
    
    long_acc = np.array(mat_data['INS'][0,0]['EKF_POS_ACC'][0,0]['Longitude_acc']).flatten()
    lat_acc = np.array(mat_data['INS'][0,0]['EKF_POS_ACC'][0,0]['Latitude_acc']).flatten()
    altitude_acc = np.array(mat_data['INS'][0,0]['EKF_POS_ACC'][0,0]['Altitude_acc']).flatten()
    precision_time = np.array(mat_data['INS'][0,0]['EKF_POS_ACC'][0,0]['t']).flatten()
    
    speed = np.array(mat_data['INS'][0,0]['EKF_VEL_BODY'][0,0]['VELOCITY_X']).flatten()
    speed_time = np.array(mat_data['INS'][0,0]['EKF_VEL_BODY'][0,0]['t']).flatten()

    # power (ams)
    p_out = np.array(mat_data['AMS'][0,0]['TsMeas'][0,0]['Pout']).flatten()
    p_time = np.array(mat_data['AMS'][0,0]['TsMeas'][0,0]['t']).flatten()
    
    # brakes
    brakeF = np.array(mat_data['PDL'][0,0]['Brakes'][0,0]['Front']).flatten()
    brakeR = np.array(mat_data['PDL'][0,0]['Brakes'][0,0]['Rear']).flatten()
    brake_time = np.array(mat_data['PDL'][0,0]['Brakes'][0,0]['t']).flatten()

    # yaw rate
    yaw_rate = np.array(mat_data['INS'][0,0]['IMU_GYRO'][0,0]['Gyro_Z']).flatten()
    yaw_time = np.array(mat_data['INS'][0,0]['IMU_GYRO'][0,0]['t']).flatten()

    # interpolation to speed_time
    lat_interp = np.interp(speed_time, gps_time, latitude)
    lon_interp = np.interp(speed_time, gps_time, longitude)
    altitude_interp = np.interp(speed_time, altitude_time, altitude)
    lat_acc_interp = np.interp(speed_time, precision_time, lat_acc)
    long_acc_interp = np.interp(speed_time, precision_time, long_acc)
    altitude_acc_interp = np.interp(speed_time, precision_time, altitude_acc)
    p_interp = np.interp(speed_time, p_time, p_out)
    brakeF_interp = np.interp(speed_time, brake_time, brakeF)
    brakeR_interp = np.interp(speed_time, brake_time, brakeR)
    yaw_rate_interp = np.interp(speed_time, yaw_time, yaw_rate)
    brake = brakeF_interp + brakeR_interp

    # filter start based on gps accuracy
    mask = (lat_acc_interp < 1.5) & (long_acc_interp < 1.5) & (altitude_acc_interp < 1.5)
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) == 0:
        raise ValueError("No data with sufficient GPS accuracy found (acc < 1.5).")
        
    start_index = valid_indices[0]
    end_index = valid_indices[-1]

    return {
        'Speed': speed[start_index:end_index], 
        'Latitude': lat_interp[start_index:end_index],
        'Longitude': lon_interp[start_index:end_index], 
        'Altitude': altitude_interp[start_index:end_index],
        'Power': p_interp[start_index:end_index],
        'Brake': brake[start_index:end_index], 
        'Time': speed_time[start_index:end_index],
        'YawRate': yaw_rate_interp[start_index:end_index] 
    }