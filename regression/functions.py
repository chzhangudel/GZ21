import numpy as np

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
    Return the BZ parameterization with the multiplicative coefficient set to
    one.
    """
    zeta = velocity['vsurf'].diff(dim='xu_ocean') - velocity['usurf'].diff(
        dim='yu_ocean')
    d = velocity['usurf'].diff(dim='yu_ocean') + velocity['v'].diff(
        dim='xu_ocean')
    d_tilda = velocity['usurf'].diff(dim='xu_ocean') - velocity['vsurf'].diff(
        dim='yu_ocean')
    zeta_sq = zeta**2
    s_x = ((zeta_sq - zeta * d).diff(dim='xu_ocean') +
           (zeta * d_tilda).diff(dim='yu_ocean'))
    s_y = (zeta * d_tilda).diff(dim='xu_ocean') + \
          (zeta_sq + zeta * d).diff(dim='yu_ocean')
    return s_x, s_y

