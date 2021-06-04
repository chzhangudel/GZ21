import numpy as np
from data.pangeo_catalog import get_grid

grid = get_grid()

# This defines standard functions used for sparse regression

class FunctionsForRegression:
    def __init__(self, lon_name: str = 'xu_ocean', lat_name: str = 'yu_ocean',
                 u_name : str = 'usurf', v_name: str = 'vsurf'):
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.u_name = u_name
        self.v_name = v_name
        self.field = None

    @property
    def field(self):
        if self._field is not None:
            return self._field
        else:
            raise ValueError('No field defined.')

    @field.setter
    def field(self, value):
        self._field = value

    def dx(self, field):
        return field.diff(dim=self.lon_name)

    def dy(self, field):
        return field.diff(dim=self.lat_name)

    def u(self):
        return self.field[self.u_name]

    def v(self):
        return self.field[self.v_name]

    def zeta(self):
        return self.dx(self.v) - self.dy(self.u)

    def sigma(self):
        return self.dx(self.u) + self.dy(self.v)

    def d(self):
        return self.dy(self.u) + self.dx(self.v)

    def d_tilda(self):
        return self.dx(self.u) - self.dy(self.v)


def bz(velocity: np.ndarray):
    """
    Return the BZ parameterization
    """
    velocity = velocity / 10
    zeta = (velocity['vsurf'].diff(dim='xu_ocean') / grid['dxt']
           - velocity['usurf'].diff(dim='yu_ocean') / grid['dyt'])
    d = (velocity['usurf'].diff(dim='yu_ocean') / grid['dyt']
        + velocity['vsurf'].diff(dim='xu_ocean') / grid['dxt'])
    d_tilda = (velocity['usurf'].diff(dim='xu_ocean') / grid['dxt']
              - velocity['vsurf'].diff(dim='yu_ocean') / grid['dyt'])
    zeta_sq = zeta**2
    s_x = ((zeta_sq - zeta * d).diff(dim='xu_ocean') / grid['dxt']
            + (zeta * d_tilda).diff(dim='yu_ocean') / grid['dyt'])
    s_y = ((zeta * d_tilda).diff(dim='xu_ocean') / grid['dxt']
          + (zeta_sq + zeta * d).diff(dim='yu_ocean') / grid['dyt'])
    k_bt = -4.87 * 1e8
    s_x, s_y = s_x * 1e7 * k_bt, s_y * 1e7 * k_bt
    print(s_x.std())
    return s_x, s_y

