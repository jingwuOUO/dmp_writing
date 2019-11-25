# if do not set the horizon plane
        SetPlane = False
        if SetPlane == True:
            q = np.array([-0.999997, -0.002155, 0.00076, -0.000308])
            q1_2 = pow(q[1], 2)
            q2_2 = pow(q[2], 2)
            q3_2 = pow(q[3], 2)
            q1_q2 = q[1] * q[2]
            q1_q3 = q[1] * q[3]
            q2_q3 = q[2] * q[3]
            q0_q1 = q[0] * q[1]
            q0_q2 = q[0] * q[2]
            q0_q3 = q[0] * q[3]
            position = np.array([-0.296881, -0.884167, -0.641915])
            Rotation = np.array([
                                [1-2*q2_2-2*q3_2, 2*q1_q2+2*q0_q3, 2*q1_q3-2*q0_q2], 
                                [2*q1_q2-2*q0_q3, 1-2*q1_2-2*q3_2, 2*q2_q3+2*q0_q1], 
                                [2*q1_q3+2*q0_q2, 2*q2_q3-2*q0_q2, 1-2*q1_q3-2*q2_2]
                                ])
            Rotation_inv = np.linalg.inv(Rotation)
            # print("rotation", Rotation)
            # print("values before", values.shape)
            values = np.subtract(values, position)
            # print("values", values.shape)
            values = np.array([np.dot(Rotation, values[i,]) for i in range(Len)])
            # print("values", values)