import numpy as np
import scipy.io

class RIS_MISO(object):
    def __init__(self,
                 num_antennas,
                 num_RIS_elements,
                 num_users,
                 channel_est_error=False,
                 AWGN_var=-169,
                 BW = 240000,
                 channel_noise_var=1e-2):

        self.M = num_antennas
        self.N = num_RIS_elements
        # self.K = num_users

        self.channel_est_error = channel_est_error

        # assert self.M == self.K #as get AssertionError when K =1
        self.BW = BW
        
        #self.awgn_var = AWGN_var
        P_noise = 10**((AWGN_var + 10*np.log10(self.BW))/10)
        self.h_normalize = np.sqrt(P_noise)
        self.awgn_var = P_noise/(self.h_normalize**2)
        
        self.channel_noise_var = channel_noise_var

        power_size = 2 * 2 * self.M

        channel_size = 2 * (self.M * self.N + self.N * 2 + 2 * self.M)

        self.action_dim = 2 * self.M * 2 + 2 * self.N #need recalculate
        self.state_dim = power_size + channel_size + self.action_dim #need recalculate

        self.H_r_all = None
        self.H_d_all = None
        self.G = None

        self.W = np.eye(self.M, dtype=complex) + 1j*np.eye(self.M, dtype=complex)
        self.Phi = np.eye(self.N, dtype=complex)+ 1j*np.eye(self.N, dtype=complex)

        self.state = None
        self.done = None

        self.episode_t = None

    def _compute_He(self):
        
        H_tilde = np.zeros(shape=(self.M,0),dtype =complex)
        for k0 in range(2):
            H_r_k = self.H_r_all[:,k0].reshape(-1,1)
            temp_3 = self.G.conjugate().T@self.Phi@H_r_k
            H_tilde = np.append(H_tilde,temp_3,1)
        H_tilde = H_tilde + self.H_d_all
        He = self.W.conjugate().T@H_tilde
        return He
    
    def _channel_generation(self):
        M = self.M       #so luong antenna of BS
        N = self.N     #so luong phan tu phan xa tren RIS
        K = 2       #number of user, not use
        
        T0 = -30    #pathloss reference in db
        d0 = 1      #reference distance in m
        alpha_BS_IRS = 2.2 #path loss reference from BS to IRS
        alpha_IRS_UE = 2.2 #path loss reference from IRS to UE
        alpha_BS_UE = 3.5 #path loss reference from BS to UE

        delta_bs=1/2  #khoang cach giua cac phan tu anten tren BS
        delta_irs=1/8 #khoang cach giua cac phan tu anten tren IRS

        K_IRS_UE = 2
        K_BS_IRS = 3 #Rician factors cho kenh tu BS den IRS

        x_bs = 0; y_bs = 0; z_bs=10 #toa do tram BS
        x_irs = 0; y_irs = 100; z_irs = 10 # toa do RIS

        #tinh toa do user = ue[0] va nghe len ue[1]
        z_ue = np.array([[1.5], [1.5]])  # z coordinates for both users
        central_points = np.array([[100, 30], [350, -70]])  # Central points for both users
        central_radii = np.array([10, 10])  # Central radii for both users
        # Generate random locations for users
        r_loc = np.random.rand(2, 1) * central_radii.reshape(-1, 1)
        # print(r_loc)
        theta_loc = np.random.rand(2, 1) * 2 * np.pi
        temp_UE_x = r_loc * np.cos(theta_loc)
        temp_UE_y = r_loc * np.sin(theta_loc)

        # Calculate x, y coordinates for both users
        x_ue = central_points[:, 0].reshape(-1, 1) + temp_UE_x
        y_ue = central_points[:, 1].reshape(-1, 1) + temp_UE_y

        # print('toa do:',x_ue, y_ue, z_ue)
        
        #generate pathloss from BS to UE       
        d_BS_UE = np.zeros(shape=(2,1))
        PL_BS_UE = np.zeros(shape=(2,1))
        for k0 in range(2):
            d_BS_UE[k0] = np.sqrt((x_ue[k0]-x_bs)**2+(y_ue[k0]-y_bs)**2+(z_ue[k0]-z_bs)**2)
            PL_BS_UE[k0]=(10**(T0/10))*(d_BS_UE[k0]/d0)**(-alpha_BS_UE)
        #print(d_BS_UE)
        #generate pathloss from BS to IRS
        d_BS_IRS = np.sqrt((x_bs-x_irs)**2+(y_bs-y_irs)**2+(z_bs-z_irs)**2)
        PL_BS_IRS = (10**(T0/10))*(d_BS_IRS/d0)**(-alpha_BS_IRS)
        #generate pathloss from IRS to UE
        d_IRS_UE = np.zeros(shape=(2,1))
        PL_IRS_UE = np.zeros(shape=(2,1))
        for k0 in range(2): 
            d_IRS_UE[k0] = np.sqrt((x_ue[k0]-x_irs)**2+(y_ue[k0]-y_irs)**2+(z_ue[k0]-z_irs)**2)
            PL_IRS_UE[k0]=(10**(T0/10))*(d_IRS_UE[k0]/d0)**(-alpha_IRS_UE)

        #generate channel from BS to IRS
        #nLOS path
        nLOS_BS_IRS = np.sqrt(0.5)*(np.random.randn(self.N,self.M)+1j*np.random.randn(self.N,self.M))
        #LOS path
        sin_angle_BS=np.sin(np.random.rand(1,1)*2*np.pi)
        sin_angle_IRS=np.sin(np.random.rand(1,1)*2*np.pi)

        steer_BS = np.zeros(shape=(1,self.M),dtype =complex)
        steer_IRS = np.zeros(shape=(self.N,1),dtype =complex)
        for m0 in range(self.M):
            steer_BS[0,m0] = np.exp(-1j*(m0)*np.pi*delta_bs*sin_angle_BS)
        for n0 in range(self.N):
            steer_IRS[n0] = np.exp(1j*(n0)*np.pi*delta_irs*sin_angle_IRS)
        LOS_BS_IRS= steer_IRS*steer_BS

        G = np.sqrt(PL_BS_IRS)*(np.sqrt(K_BS_IRS/(K_BS_IRS+1))*LOS_BS_IRS+np.sqrt(1/(K_BS_IRS+1))*nLOS_BS_IRS);

        #generate channel from BS to UE
        #nLOS path
        nLOS_BS_UE_all = np.zeros(shape=(self.M,0))

        for k0 in range(2):
            temp_1=np.sqrt(PL_BS_UE[k0])*np.sqrt(0.5)*(np.random.randn(self.M,1)+1j*np.random.randn(self.M,1))
            nLOS_BS_UE_all = np.append(nLOS_BS_UE_all,temp_1,1)
        # H_d_all = nLOS_BS_UE_all.tolist()
        H_d_all = nLOS_BS_UE_all

        #generate channel from IRS to UE
        H_r_all= np.zeros(shape=(self.N,0),dtype =complex)
        LOS_IRS_UE_temp = np.zeros(shape=(self.N,1),dtype =complex)

        for k0 in range(2):
                #nLOS path
            nLOS_IRS_UE_temp=np.sqrt(0.5)*(np.random.randn(self.N,1)+1j*np.random.randn(self.N,1))
                #LOS path
            sin_angle_UE_temp=np.sin(np.random.rand(1,1)*2*np.pi)
            for n0 in range(self.N):
                LOS_IRS_UE_temp[n0]=np.exp(-1j*(n0)*np.pi*delta_irs*sin_angle_UE_temp)
            temp_2= np.sqrt(PL_IRS_UE[k0])*(np.sqrt(K_IRS_UE/(K_IRS_UE+1))*LOS_IRS_UE_temp+np.sqrt(1/(K_IRS_UE+1))*nLOS_IRS_UE_temp)
            H_r_all = np.append(H_r_all,temp_2,1)
        G = G /np.sqrt(self.h_normalize)
        H_r_all = H_r_all/np.sqrt(self.h_normalize)
        H_d_all = H_d_all/self.h_normalize
        # print("type G[0], H_r, H_d",type(G[0]), type(H_r_all), type(H_d_all))
        return G,H_r_all,H_d_all

    # def reset(self,predict_mode,episode_num):
    def reset(self,predict_mode,episode_num, G, H_r_all, H_d_all):
        self.episode_t = 0
        self.G = G
        self.H_r_all = H_r_all
        self.H_d_all = H_d_all
        # self.G, self.H_r_all, self.H_d_all = self._channel_generation()
        # if predict_mode:
        #     np.save(f'./channel/N_{self.N}_channel_{episode_num}.npy', {'G': self.G, 'H_r_all': self.H_r_all, 'H_d_all': self.H_d_all})
            # scipy.io.savemat(f'./channel/channel_{episode_num}.mat',{'G':self.G , 'H_r_all':self.H_r_all , 'H_d_all':self.H_d_all})
        self.W = np.eye(self.M, dtype=complex) + 1j*np.eye(self.M, dtype=complex)
        self.Phi = np.eye(self.N, dtype=complex)+ 1j*np.eye(self.N, dtype=complex)

        init_action_W = np.hstack((np.real(self.W.reshape(1, -1)), np.imag(self.W.reshape(1, -1))))
        init_action_Phi = np.hstack((np.real(np.diag(self.Phi)).reshape(1, -1), np.imag(np.diag(self.Phi)).reshape(1, -1)))

        init_action = np.hstack((init_action_W, init_action_Phi))

 
        He = self._compute_He()
        He_real, He_img = np.real(He).reshape(1, -1), np.imag(He).reshape(1, -1)
        G_real, G_imag = np.real(self.G).reshape(1, -1), np.imag(self.G).reshape(1, -1)
        H_r_real, H_r_imag = np.real(self.H_r_all).reshape(1, -1), np.imag(self.H_r_all).reshape(1, -1)
        H_d_real, H_d_imag = np.real(self.H_d_all).reshape(1, -1), np.imag(self.H_d_all).reshape(1, -1)

        self.state = np.hstack((init_action, He_real, He_img, G_real, G_imag ,H_r_real, H_r_imag, H_d_real, H_d_imag))

        return self.state
    def _compute_reward(self):
        reward = 0
        opt_reward = 0
        He = self._compute_He()
        He = np.abs(He)**2
        # SINR = np.zeros(shape=(2,1)) #not use yet
        SINR_opt = np.zeros(shape=(2,1))
        for k0 in range(2):
            tmp = He[:,k0]
            # SINR[k0] = tmp[k0]/(np.sum(tmp)-tmp[k0]+self.awgn_var) #not use yet
            SINR_opt[k0] = tmp[k0]/self.awgn_var

        # reward = np.sum(np.log2(1+SINR))
        # opt_reward = np.sum(np.log2(1+SINR_opt)) #tuan's thesis

        reward = np.float64(np.log2(1+SINR_opt[0]) - np.log2(1+SINR_opt[1]))
        opt_reward = np.float64(np.log2(1+SINR_opt[0]))
        # print(type(reward), type(opt_reward))
        return reward, opt_reward
    
    def step(self, action):
        self.episode_t += 1

        action = action.reshape(1, -1)

        W_real = action[:, :self.M ** 2]
        W_imag = action[:, self.M ** 2:2 * self.M ** 2]

        Phi_real = action[:, -2 * self.N:-self.N]
        Phi_imag = action[:, -self.N:]

        self.W = W_real.reshape(self.M, 2) + 1j * W_imag.reshape(self.M, 2)
        self.Phi = np.eye(self.N, dtype=complex) * (Phi_real + 1j * Phi_imag)
        #print(self.Phi)
        He = self._compute_He()        
        He_real, He_img = np.real(He).reshape(1, -1), np.imag(He).reshape(1, -1)
        G_real, G_imag = np.real(self.G).reshape(1, -1), np.imag(self.G).reshape(1, -1)
        H_r_real, H_r_imag = np.real(self.H_r_all).reshape(1, -1), np.imag(self.H_r_all).reshape(1, -1)
        H_d_real, H_d_imag = np.real(self.H_d_all).reshape(1, -1), np.imag(self.H_d_all).reshape(1, -1)

        self.state = np.hstack((action,He_real, He_img, G_real, G_imag, H_r_real, H_r_imag, H_d_real, H_d_imag))

        reward, opt_reward = self._compute_reward() 

        done = opt_reward == reward

        return self.state, reward, done, None

    def close(self):
        pass
