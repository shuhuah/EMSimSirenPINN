import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
import math
from scipy.ndimage import gaussian_filter





def generate_permittivity_model(n1, n2, center_diam = 20, inclusion_value=1.0):

    perm_model = np.ones((n1, n2))

    center = (n1 // 2, n2 // 2)
    for i in range(n1):
        for j in range(n2):
            if np.linalg.norm(np.array([i, j]) - np.array(center)) < center_diam / 2:
                perm_model[i, j] = inclusion_value
    
    if inclusion_value != 1.0:
        # Apply a smoothing to the model
        perm_model = gaussian_filter(perm_model, sigma=2.0)

    return perm_model



def generate_boundary_coeff(n1, n2, npml, f, f0):
    """
    Calculates Perfectly Matched Layer (PML) coefficients for wave modeling.

    The value for a0 is taken from Y. Zeng et al., "The application of the 
    perfectly matched layer in numerical modeling of wave propagation in 
    poroelastic media," Geophysics, (66) 2001, 1258-1266.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
        three complex PML coefficient arrays (pmla, pmlb, pmlc).
    """
    if n1<=2*npml or n2<=2*npml:
        raise ValueError("PML thickness is too large for the given grid size.")

    # This value is from Zeng et al ., 2001, Geophysics
    a0 = 1.79

    # --- 1. Calculate total grid dimensions including PML regions ---
    Nz = n2
    Nx = n1

    # --- 2. Create PML decay profiles (lx and lz) using vectorization ---
    # Create the horizontal decay profile (lx)
    lx_row = np.zeros(Nx)
    if npml > 0:
        ramp_down = np.arange(npml, 0, -1)
        ramp_up = np.arange(1, npml + 1)
        lx_row[:npml] = ramp_down
        lx_row[-npml:] = ramp_up
    # Tile the row to create the full 2D array
    lx = np.tile(lx_row.reshape(-1, 1), (1, Nz))

    print("lx shape:", lx.shape)

    # Create the vertical decay profile (lz)
    lz_col = np.zeros(Nz)
    if npml > 0:
        ramp_down = np.arange(npml, 0, -1)
        ramp_up = np.arange(1, npml + 1)
        lz_col[:npml] = ramp_down
        lz_col[-npml:] = ramp_up
    # Tile the column (after reshaping) to create the full 2D array
    lz = np.tile(lz_col, (Nx, 1))

    print("lz shape:", lz.shape)

    # --- 3. Calculate PML coefficients ---
    coef = -1 * (f0 / f) * a0

    # Calculate the imaginary parts of the complex coefficients
    ex_imag = coef * (lx / npml)**2 if npml > 0 else np.zeros((Nx, Nz))
    ez_imag = coef * (lz / npml)**2 if npml > 0 else np.zeros((Nx, Nz))

    # Create the complex arrays. In NumPy, complex numbers are easily
    # created by adding a real part to an imaginary part (multiplied by 1j).
    ex = np.ones((Nx, Nz)) + 1j * ex_imag
    ez = np.ones((Nx, Nz)) + 1j * ez_imag

    # Calculate the final PML parameters.
    pmla = ez / ex
    pmlb = ex / ez
    pmlc = ex * ez

    return pmla, pmlb, pmlc



def generate_point_source(n1, n2, n1_s, n2_s, sigma, max_amplitude=1.0, apply_threshold=True):
    '''
    get point source, wigh Gaussian shape
    input: n1, n2, n1_s, n2_s
    output: point source array
    '''

    threshold = 0.001    # cutoff value

    # Coordinate grids
    x = np.arange(n1)[:, None]  # shape (n1, 1)
    y = np.arange(n2)[None, :]  # shape (1, n2)

    # Gaussian centered at (n1_s, n2_s) with max = max_amplitude
    dist_sq = (x - n1_s)**2 + (y - n2_s)**2
    ps = max_amplitude * np.exp(-dist_sq / (2 * sigma**2))

    # Apply threshold
    if apply_threshold:
        ps[ps < threshold] = 0.0

    return ps

def generate_mesh_grid(n1, n2, d1, d2):
    '''
    Generate a mesh grid for the simulation domain.
    '''
    x1 = np.linspace(0, d1*(n1-1), n1)
    x2 = np.linspace(0, d2*(n2-1), n2)
    print("Mesh grid shapes:", x1.shape, x2.shape)
    return np.meshgrid(x1, x2, indexing='ij')


def prepare_data_training(input):
    return input.reshape(input.shape[0] * input.shape[1], 1)




# --- Sine activation with w0 ---
class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# --- SIREN network ---
class SIREN(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features, 
                 lb, ub, 
                 w0=1.0, w0_initial=30.0,
                 DEBUG_FLAG = False):
        super().__init__()
        self.w0 = w0

        self.lb = lb
        self.ub = ub

        self.DEBUG_FLAG = DEBUG_FLAG

        layers = []
        # First layer
        first = nn.Linear(in_features, hidden_features)
        self.siren_init(first, w0_initial, first_layer=True)
        layers.append(first)
        layers.append(Sine(w0_initial))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            hidden = nn.Linear(hidden_features, hidden_features)
            self.siren_init(hidden, w0)
            layers.append(hidden)
            layers.append(Sine(w0))

        # Output layer
        final = nn.Linear(hidden_features, out_features)
        self.siren_init(final, w0)
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def siren_init(self, layer, w0, first_layer=False):
        with torch.no_grad():
            num_in = layer.in_features
            if first_layer:
                bound = 1 / num_in
            else:
                bound = math.sqrt(6 / num_in) / w0
            layer.weight.uniform_(-bound, bound)
            layer.bias.uniform_(-bound, bound)

    def forward(self, x):

        if self.DEBUG_FLAG:
            print( "In SIREN forward, x.shape", x.shape, "x[:,0].min", x[:,0].min(), "x[:,0].max", x[:,0].max(), "x[:,0].shape", x[:,0].shape )
            print("self.lb", self.lb, "self.ub", self.ub)
            print("self.lb.shape", self.lb.shape, "self.ub.shape", self.ub.shape)

        if self.lb is not None and self.ub is not None:
            x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(x)




def fwd_gradients(y, x, device, DEBUG_FLAG=False):
    """
    Forward-mode gradient trick: computes dy/dx, then the derivative of that
    w.r.t. dummy variable. Works for complex tensors in PyTorch.
    """

    if DEBUG_FLAG:
        print("In fwd_gradients:")
        print("y shape:", y.shape, "x shape:", x.shape, "y dtype", y.dtype, "x dtype", x.dtype)

    ones = torch.ones_like(y, device=device, requires_grad=True)
    g = torch.autograd.grad(y, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    if DEBUG_FLAG:
        print("g shape:", g.shape, "g dtype", g.dtype)

    y_x = torch.autograd.grad(g, ones, torch.ones_like(g), create_graph=True)[0]
    if DEBUG_FLAG:
        print("y_x shape:", y_x.shape, "y_x dtype", y_x.dtype)
    return y_x



class PINN(torch.nn.Module):
    def __init__(self, x, z, A, B, C, ps, 
                 m, omega, da_dx, db_dy,
                 lr=1e-4,
                 niter = 500,
                 lbfgs_max_iter = 500,
                 in_features=2, 
                 hidden_features=64, 
                 hidden_layers=8, 
                 out_features=2, 
                 w0=1.0, 
                 w0_initial=30.0,
                 device=None,
                 DEBUG_FLAG = False,
                 ):
        super().__init__()


        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        X = np.concatenate([x, z], 1)
        self.lb = torch.tensor(X.min(0), dtype=torch.float32, device=self.device)
        self.ub = torch.tensor(X.max(0), dtype=torch.float32, device=self.device)

        self.DEBUG_FLAG = DEBUG_FLAG

        if self.DEBUG_FLAG:
            print("self.lb", self.lb, "self.ub", self.ub)


        self.model = SIREN(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            lb=self.lb,
            ub=self.ub,
            w0=w0,
            w0_initial=w0_initial,
            DEBUG_FLAG=self.DEBUG_FLAG,
        )


        # Store training data as tensors
        self.x = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.z = torch.tensor(z, dtype=torch.float32, device=self.device)

        self.boundary_x = (self.x == 0) | (self.x == self.x.max())
        self.boundary_z = (self.z == 0) | (self.z == self.z.max())

        # Keep A, B, C, ps as complex
        self.A = torch.tensor(A, dtype=torch.complex64, device=self.device)
        self.B = torch.tensor(B, dtype=torch.complex64, device=self.device)
        self.C = torch.tensor(C, dtype=torch.complex64, device=self.device)


        self.ps = torch.tensor(ps, dtype=torch.float32, device=self.device)

        # self.source_mask = 1.0 - (self.ps != 0).to(torch.float32)

        self.source_mask = 1.0 - self.ps


        # m stays real
        self.m = torch.tensor(m, dtype=torch.float32, device=self.device)

        self.omega = omega
        self.lr = lr

        self.niter = niter
        self.lbfgs_max_iter = lbfgs_max_iter

        self.da_dx = torch.tensor(da_dx, dtype=torch.complex64, device=self.device)
        self.db_dy = torch.tensor(db_dy, dtype=torch.complex64, device=self.device)

        self.lbfgs_iter_count = 0

        if self.DEBUG_FLAG:
            print(self.da_dx.shape, self.da_dx.dtype)


    def PDE_residual(self, x, z):

        range = self.ub - self.lb
        if self.DEBUG_FLAG:
            print("range:", range)

        u = self.model(torch.cat([x, z], dim=1))
        u_real = u[:, 0:1]
        u_imag = u[:, 1:2]
        u_cmplx = torch.complex(u_real, u_imag)

        if self.DEBUG_FLAG:
            print("u_cmplx shape:", u_cmplx.shape, u_cmplx.dtype)

        if self.DEBUG_FLAG:
            print("-------------Grad---------------")
        dudx = fwd_gradients(u_cmplx, x, 
                            device=self.device, 
                            DEBUG_FLAG=self.DEBUG_FLAG)
        dudz = fwd_gradients(u_cmplx, z, 
                            device=self.device, 
                            DEBUG_FLAG=self.DEBUG_FLAG)

        if self.DEBUG_FLAG:
            print("self.A.shape", self.A.shape, "self.B.shape", self.B.shape)
            print("self.A.dtype", self.A.dtype, "self.B.dtype", self.B.dtype)
            print("dudx.shape", dudx.shape, "dudx.dtype", dudx.dtype)
            print("dudz.shape", dudz.shape, "dudz.dtype", dudz.dtype)
            print("---------------Laplacian-----------------")

        dudxx = fwd_gradients(self.A * dudx, x, 
                                device=self.device, 
                                DEBUG_FLAG=self.DEBUG_FLAG)
        dudzz = fwd_gradients(self.B * dudz, z, 
                                device=self.device, 
                                DEBUG_FLAG=self.DEBUG_FLAG)


        if self.DEBUG_FLAG:
            print("dudxx shape:", dudxx.shape, dudxx.dtype)
            print("dudzz shape:", dudzz.shape, dudzz.dtype)
            print("END")
        right_side = self.omega * self.ps
        # print("right_side shape:", right_side.shape, right_side.dtype)

        f_loss = self.C * self.omega * self.omega * self.m * u_cmplx \
            +  dudxx \
            +  dudzz - 1j * right_side
    
        return u_real, u_imag, f_loss
    
    def loss_fn(self):
        x = self.x.clone().requires_grad_(True)
        z = self.z.clone().requires_grad_(True)
        _, _, f_loss = self.PDE_residual(x, z)
        # masked_loss = f_loss * self.source_mask
        masked_loss = f_loss 
        # print("f_loss shape:", f_loss.shape, f_loss.dtype)
        return torch.sum(torch.abs(masked_loss) ** 2)
    

    def train_model(self):

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        start_time = time.time()

        misfit = []
        for epoch in range(self.niter):
            optimizer.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            misfit.append(loss.item())
            optimizer.step()
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4e}, Time: {time.time() - start_time:.2f}s")
                start_time = time.time()


        lbfgs = torch.optim.LBFGS(self.model.parameters(),
                                max_iter=self.lbfgs_max_iter, tolerance_grad=1e-12, tolerance_change=1e-12)

        def closure():
            lbfgs.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            
            misfit.append(loss.item())
            self.lbfgs_iter_count += 1
            if self.lbfgs_iter_count % 100 == 0:
                print(f"LBFGS Iteration {self.lbfgs_iter_count}, Loss: {loss.item():.4e}")
            return loss
        
        lbfgs.step(closure)

        return misfit

    def predict(self, x, z):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
            u = self.model(torch.cat([x, z], dim=1))
            u_real = u[:, 0:1]
            u_imag = u[:, 1:2]
            return u_real.cpu().numpy(), u_imag.cpu().numpy()
