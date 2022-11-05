import matplotlib.pyplot as plt
import numpy as np
import math
import os


class Surfaces_math_functions:

    def __init__(self, X_MIN, X_MAX, Y_MIN, Y_MAX, COFF, OFFSET, equation_name):
        self.X_MIN = X_MIN
        self.X_MAX = X_MAX
        self.Y_MIN = Y_MIN
        self.Y_MAX = Y_MAX
        self.COFF = COFF
        self.OFFSET = OFFSET
        self.equation_name = equation_name
        self.equation = self.__get_equation(equation_name)

    # ---- Private methods ----

    # Private getters
    def __get_equation(self, string):
        string = string.lower()
        equation_table = {
            'paraboloide hiperbolico' or 'silla de montura':
                [
                    lambda x, y: ((y**2)/5 - (x**2)/3) * self.COFF,
                    "y^2/5 - x^2/3"
                ],
            'seno' or 'sin':
                [
                    lambda x, y: (np.sin(x+y)) * self.COFF,
                    "sin(x+y)"
                ],
            'coseno' or 'cos':
                [
                    lambda x, y: (np.cos(x+y)) * self.COFF,
                    "cos(x+y)"
                ],
            'tangente' or 'tan':
                [
                    lambda x, y: (np.tan(x+y)) * self.COFF,
                    "tan(x+y)"
                ],
            'exponencial' or 'exp':
                [
                    lambda x, y: (np.exp(x+y)) * self.COFF,
                    "e^(x+y)"
                ],
            'logaritmo' or 'log':
                [
                    lambda x, y: (np.log(x**2+y**2+1)) * self.COFF,
                    "log(x^2+y^2+1)"
                ],
            'raiz cuadrada' or 'sqrt':
                [
                    lambda x, y: (np.sqrt(x**2+y**2)) * self.COFF,
                    "sqrt(x^2+y^2)"
                ],
            'seno hiperbolico' or 'sinh':
                [
                    lambda x, y: (np.sinh(x+y)) * self.COFF,
                    "sinh(x+y)"
                ],
            'coseno hiperbolico' or 'coseno hiperbolico' or 'cosh':
                [
                    lambda x, y: (np.cosh(x+y)) * self.COFF,
                    "cosh(x+y)"
                ],
            'superficie 1' or 's1':
                [
                    lambda x, y: ((1-x/2+x**5+y**3) *
                                  np.exp(-x**2-y**2)) * self.COFF,
                    "(1-x/2+x^5+y^3) * e^(-x^2-y^2)"
                ],
            'superficie 2' or 's2':
                [
                    lambda x, y: ((x**2+y**2)*np.exp(-x**2-y**2)) * self.COFF,
                    "(x^2+y^2) * e^(-x^2-y^2)"
                ],
            'superficie 3' or 's3':
                [
                    lambda x, y: ((x**2+y**2)/(x**2+y**4)) * self.COFF,
                    "(x^2+y^2)/(x^2+y^4)"
                ],
            'superficie 4' or 's4':
                [
                    lambda x, y: ((x**2*y)/(x**2+y**4)) * self.COFF,
                    "(x^2*y)/(x^2+y^4)"
                ],
            'superficie 5' or 's5':
                [
                    lambda x, y: ((x**2*y**2)/(x**2+y**4)) * self.COFF,
                    "(x^2*y^2)/(x^2+y^4)"
                ],
        }

        return equation_table[string]

    def __get_surface_image(self):
        x = np.arange(self.X_MIN, self.X_MAX, 0.6)
        y = np.arange(self.Y_MIN, self.Y_MAX, 0.6)
        X, Y = np.meshgrid(x, y)
        Z = self.equation[0](X, Y)
        # reshaped_Z = np.reshape(Z, (len(x), len(y)))
        return X, Y, Z

    # ---- Public methods ----

    # Public getters

    def get_plot_data(self):
        X, Y, Z = self.__get_surface_image()
        return X, Y, Z

    def get_equation_function(self):
        return self.equation[0]

    def get_cartesian_equation_str(self):
        return self.equation[1]


class Fan_mapping:

    def __init__(self, QTY_FAN, V_FAN, I_FAN, surface_eq):
        self.QTY_FAN = QTY_FAN
        self.ROWS_FAN, self.COLS_FAN = self.__get_row_col()
        self.V_FAN = V_FAN
        self.I_FAN = I_FAN
        self.surface_eq = surface_eq

    # ---- Private methods ----

    # Create a matrix given the fan's quantity
    def __fan_to_grid(self):
        # 1.01 to avoid 0 (common discontinuity)
        _x = np.arange(-self.ROWS_FAN/2, self.ROWS_FAN/2, 1.01)
        # 1.01 to avoid 0 (common discontinuity)
        _y = np.arange(-self.COLS_FAN/2, self.COLS_FAN/2, 1.01)
        _xx, _yy = np.meshgrid(_x, _y)
        X, Y = _xx.ravel(), _yy.ravel()
        return X, Y

    # Private getters
    def __get_row_col(self):
        rows = math.ceil(math.sqrt(self.QTY_FAN))
        cols = math.ceil(self.QTY_FAN / rows)
        return rows, cols

    def __get_voltage_point(self, X, Y):

        voltage = self.surface_eq.get_equation_function()(X, Y)

        # Normalize values
        self.max_value = np.amax(voltage)
        self.min_value = np.amin(voltage)
        self.mean_value = np.mean(voltage)
        voltage = (voltage - self.min_value) / \
            (self.max_value - self.min_value)
        return voltage * self.V_FAN

    def __get_voltage_map(self):
        X, Y = self.__fan_to_grid()
        Z = self.__get_voltage_point(X, Y)
        self.max_value = round(np.max(Z), 2)
        self.min_value = round(np.min(Z), 2)
        self.mean_value = round(np.mean(Z), 2)
        self.power_consumption = round(np.sum(Z * self.I_FAN), 2)
        return X, Y, Z

    # ---- Public methods ----

    # Public getters

    def get_max_value(self):
        return self.max_value

    def get_min_value(self):
        return self.min_value

    def get_mean_value(self):
        return self.mean_value

    def get_power_consumption(self):
        return self.power_consumption

    def get_plot_data(self):
        X, Y, Z = self.__get_voltage_map()
        return X, Y, Z

    def get_row_col(self):
        return self.ROWS_FAN, self.COLS_FAN


def input_data():

    # TODO Check if the limits are correct

    qty_fan = int(input("¿Cuantos ventiladores hay disponibles? "))
    if qty_fan <= 1:
        raise ValueError("La cantidad de ventiladores debe ser mayor a 0")

    v_fan = float(input("¿Cuál es la tension nominal de los ventiladores? "))
    if v_fan <= 1:
        raise ValueError("La tension de los ventiladores debe ser mayor a 0")

    i_fan = float(input("¿Cuál es la corriente nominal de los ventiladores? "))
    if i_fan <= 0:
        raise ValueError("La corriente de los ventiladores debe ser mayor a 0")

    return qty_fan, v_fan, i_fan


def main():

    # ---- General parameters ----
    eq_available = ["paraboloide hiperbolico",
                    # "superficie 1",
                    # "superficie 2",
                    # "superficie 3",
                    # "superficie 4",
                    # "superficie 5",
                    # "seno",
                    # "coseno",
                    # "tangente",
                    # "exponencial",AAAz
                    
                    # "logaritmo",
                    # "raiz cuadrada",
                    # "seno hiperbolico",
                    # "coseno hiperbolico"
                    ]

    X_MIN = -3
    X_MAX = 3
    Y_MIN = -3
    Y_MAX = 3
    COFF = 1
    OFFSET = 0

    # ---- Input data ----
    qty_fan, v_fan, i_fan = input_data()
    # qty_fan, v_fan, i_fan = 50, 12, 0.3

    # ---- Folder creation ----
    folder_name = "./" + str(qty_fan) + " ventiladores"
    if not os.path.exists(folder_name):

        os.makedirs(folder_name)

    # General parameters log
    with open(folder_name + "/parametros.txt", "w") as f:
        f.write("Cantidad de ventiladores: " + str(qty_fan) + "\n")
        f.write("Tension nominal de los ventiladores: " + str(v_fan) + "\n")
        f.write("Corriente nominal de los ventiladores: " + str(i_fan) + "\n")
        f.write("X_MIN: " + str(X_MIN) + "\n")
        f.write("X_MAX: " + str(X_MAX) + "\n")
        f.write("Y_MIN: " + str(Y_MIN) + "\n")
        f.write("Y_MAX: " + str(Y_MAX) + "\n")
        f.write("COFF: " + str(COFF) + "\n")
        f.write("OFFSET: " + str(OFFSET) + "\n")

    figs_pairs = []
    for i in range(len(eq_available)):
        figs_pairs.append(plt.figure(figsize=(8, 4)))
        figs_pairs[i].suptitle(eq_available[i], fontsize=12)
        figs_pairs[i].subplots_adjust(hspace=0.5)
        figs_pairs[i].subplots_adjust(wspace=0.5)
        figs_pairs[i].subplots_adjust(
            left=0.11, right=0.85, top=0.9, bottom=0.3)

    # Generate plot for each surface-fan_voltage_map pair
    for i, eq_str in enumerate(eq_available):

        # Instantiate surface equation
        surface_eq = Surfaces_math_functions(
            X_MIN, X_MAX, Y_MIN, Y_MAX, COFF, OFFSET, eq_str)

        # Instantiate fan_voltage_map
        fan_map = Fan_mapping(QTY_FAN=qty_fan, V_FAN=v_fan,
                              I_FAN=i_fan, surface_eq=surface_eq)

        # Surface plot data
        x_s, y_s, z_s = surface_eq.get_plot_data()
        # Voltage map data
        x_f, y_f, z_f = fan_map.get_plot_data()

        # Plot surface
        surface_plot = figs_pairs[i].add_subplot(1, 2, 1, projection='3d')
        surface_plot.plot_surface(
            x_s, y_s, z_s, rstride=1, cstride=1, cmap='inferno', edgecolor='none')
        surface_plot.set_title(
            "f(x,y)= " + surface_eq.get_cartesian_equation_str())
        surface_plot.set_xlabel("X")
        surface_plot.set_ylabel("Y")
        surface_plot.set_zlabel("Z")
        surface_plot.set_xlim3d(X_MIN, X_MAX)
        surface_plot.set_ylim3d(Y_MIN, Y_MAX)

        # Plot fan voltage map
        fan_plot = figs_pairs[i].add_subplot(1, 2, 2, projection='3d')
        fan_plot.bar3d(x_f, y_f, 0, 1, 1, z_f, shade=True, color='red')
        fan_plot.set_title("Mapa de ventiladores")
        fan_plot.set_xlabel("X")
        fan_plot.set_ylabel("Y")
        fan_plot.set_zlabel("Z")
        fan_plot.set_xlim3d(-fan_map.ROWS_FAN/2, fan_map.ROWS_FAN/2)
        fan_plot.set_ylim3d(-fan_map.COLS_FAN/2, fan_map.COLS_FAN/2)
        fan_plot.set_zlim3d(0, fan_map.V_FAN)

        #Labels (V_max, V_min, P_total)
        figs_pairs[i].text(0.3, 0.15, "V_max: " + str(fan_map.get_max_value()) +
                           " V", ha='center', va='center', transform=figs_pairs[i].transFigure)
        figs_pairs[i].text(0.5, 0.15, "V_min: " + str(fan_map.get_min_value()) +
                           " V", ha='center', va='center', transform=figs_pairs[i].transFigure)
        figs_pairs[i].text(0.7, 0.15, "V_mean: " + str(fan_map.get_mean_value()) +
                           " V", ha='center', va='center', transform=figs_pairs[i].transFigure)
        figs_pairs[i].text(0.5, 0.05, "P_total: " + str(fan_map.get_power_consumption()) +
                           " W", ha='center', va='center', transform=figs_pairs[i].transFigure)

        # Save figure
        figs_pairs[i].savefig(folder_name + "/" + eq_str + ".png")

        print("x_f: " + str(x_f))
        print("y_f: " + str(y_f))
        print("z_f: " + str(z_f))

        #Write in file the fan_voltage_map as lookup table (for the Arduino)
        with open(folder_name + "/" + eq_str + ".txt", "w") as f:
            rows,cols = fan_map.get_row_col()
            f.write("const static float fan_voltage_map[" + str(rows) + "][" + str(cols) + "] = {")
            # for i in range(rows):
            #     for j in range(cols):
            #         f.write(str(fan_map.get_value(i,j)) + ",")
            #     f.write("\n")
            f.write("};")

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
