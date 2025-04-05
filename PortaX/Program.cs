using System;
using System.Runtime.CompilerServices;

Console.WriteLine("Porta XOR - Redes Neurais");

double[,] matrizXor = new double[,]
{
            { 0, 0, 0 },
            { 0, 1, 1 },
            { 1, 0, 1 },
            { 1, 1, 0 }
};

const int epocas = 100000;
const double n = 0.1;

//Valores primeira epoca

double w11 = 0.50;
double w21 = 0.30;
double b1 = 0.20;

double w12 = 0.40;
double w22 = 0.60;
double b2 = 0.10;

double v1 = 0.30;
double v2 = 0.40;
double b3 = 1;

for (int epoca = 1; epoca <= epocas; epoca++)
{
    Console.WriteLine("--x--");
    Console.WriteLine($"Epoca nro: {epoca}");

    var finalOutput = new double[4];
    var constFuncions = new double[4];
    var errorDerivates = new double[4];
    var errorDerivatesH1 = new double[4];
    var errorDerivatesH2 = new double[4];
    var outputErrorGradients = new double[4];
    var hiddenErrorGradientsH1 = new double[4];
    var hiddenErrorGradientsH2 = new double[4];
    var outputWeightChangesV1 = new double[4];
    var outputWeightChangesV2 = new double[4];

    var hiddenWeightChangesW11 = new double[4];
    var hiddenWeightChangesW12 = new double[4];
    var hiddenWeightChangesW21 = new double[4];
    var hiddenWeightChangesW22 = new double[4];

    var outputWeightGradientDerivationV1 = new double[4];
    var outputWeightGradientDerivationV2 = new double[4];
    var hiddenWeightGradientDerivationW11 = new double[4];
    var hiddenWeightGradientDerivationW12 = new double[4];
    var hiddenWeightGradientDerivationW21 = new double[4];
    var hiddenWeightGradientDerivationW22 = new double[4];

    var v1_new = new double[4];
    var v2_new = new double[4];

    var w11_new = new double[4];
    var w12_new = new double[4];
    var w21_new = new double[4];
    var w22_new = new double[4];

    var b1_new = new double[4];
    var b2_new = new double[4];
    var b3_new = new double[4];

    for (int amostra = 0; amostra <= 3; amostra++)
    {
        var x1 = matrizXor[amostra, 0];
        var x2 = matrizXor[amostra, 1];
        var y = matrizXor[amostra, 2];// real (esperado)

        var z1 = w11 * x1 + w21 * x2 + b1;
        var h1 = sigmoide(z1);

        var z2 = w12 * x1 + w22 * x2 + b2;
        var h2 = sigmoide(z2);

        var z3 = v1 * h1 + v2 * h2 + b3;

        var yy = sigmoide(z3); // predito

        finalOutput[amostra] = yy;

        var e = Math.Pow(y - yy, 2) / 2; //erro
        constFuncions[amostra] = e;

        errorDerivates[amostra] = yy * (1 - yy);
        errorDerivatesH1[amostra] = h1 * (1 - h1);
        errorDerivatesH2[amostra] = h2 * (1 - h2);

        var output = -2 * (y - yy) * yy * (1 - yy);

        outputErrorGradients[amostra] = output;
        hiddenErrorGradientsH1[amostra] = outputErrorGradients[amostra] * v1 * h1 * (1 - h1);
        hiddenErrorGradientsH2[amostra] = outputErrorGradients[amostra] * v2 * h2 * (1 - h2);

        outputWeightGradientDerivationV1[amostra] = outputErrorGradients[amostra] * h1;
        outputWeightGradientDerivationV2[amostra] = outputErrorGradients[amostra] * h2;
        hiddenWeightGradientDerivationW11[amostra] = hiddenErrorGradientsH1[amostra] * x1;
        hiddenWeightGradientDerivationW12[amostra] = hiddenErrorGradientsH2[amostra] * x1;
        hiddenWeightGradientDerivationW21[amostra] = hiddenErrorGradientsH1[amostra] * x2;
        hiddenWeightGradientDerivationW22[amostra] = hiddenErrorGradientsH2[amostra] * x2;

        outputWeightChangesV1[amostra] = n * outputWeightGradientDerivationV1[amostra];
        outputWeightChangesV2[amostra] = n * outputWeightGradientDerivationV2[amostra];
        hiddenWeightChangesW11[amostra] = n * hiddenWeightGradientDerivationW11[amostra];
        hiddenWeightChangesW12[amostra] = n * hiddenWeightGradientDerivationW12[amostra];
        hiddenWeightChangesW21[amostra] = n * hiddenWeightGradientDerivationW21[amostra];
        hiddenWeightChangesW22[amostra] = n * hiddenWeightGradientDerivationW22[amostra];

        v1_new[amostra] = v1 - outputWeightChangesV1[amostra];
        v2_new[amostra] = v2 - outputWeightChangesV2[amostra];

        w11_new[amostra] = w11 - hiddenWeightChangesW11[amostra];
        w12_new[amostra] = w12 - hiddenWeightChangesW12[amostra];
        w21_new[amostra] = w21 - hiddenWeightChangesW21[amostra];
        w22_new[amostra] = w22 - hiddenWeightChangesW22[amostra];

        b1_new[amostra] = b1 - n * hiddenErrorGradientsH1[amostra];
        b2_new[amostra] = b2 - n * hiddenErrorGradientsH2[amostra];
        b3_new[amostra] = b3 - n * outputErrorGradients[amostra];

        v1 = v1_new[amostra];
        v2 = v2_new[amostra];

        w11 = w11_new[amostra];
        w12 = w12_new[amostra];
        w21 = w21_new[amostra];
        w22 = w22_new[amostra];

        b1 = b1_new[amostra];
        b2 = b2_new[amostra];
        b3 = b3_new[amostra];

    }

    if ((epoca == epocas) || (constFuncions[0] <= 0.02) && (constFuncions[1] <= 0.02) && (constFuncions[2] <= 0.02) && (constFuncions[3] <= 0.02))
    {
        Console.WriteLine("");
        Console.WriteLine($"***Porta XOR - Backpropagation - Erro aceitável encontrado na época: {epoca}/{epocas}");
        Console.WriteLine("");

        for (int i = 0; i <= 3; i++)
        {
            Console.WriteLine($"Amostra {i} = x1 -> {matrizXor[i, 0]} x2 -> {matrizXor[i, 1]}  Real(y) -> {matrizXor[i, 2]}  Predito(yy) -> {finalOutput[i]:F4} Erro -> {constFuncions[i]:F4}");
        }


        Console.WriteLine("");
        Console.WriteLine($"Pesos resultantes do modelo após {epoca} epocas:");
        Console.WriteLine("");
        Console.WriteLine($"v1 : {v1,10:F4}");
        Console.WriteLine($"v2 : {v2,10:F4}");
        Console.WriteLine($"w11: {w11,10:F4}");
        Console.WriteLine($"w12: {w12,10:F4}");
        Console.WriteLine($"w21: {w21,10:F4}");
        Console.WriteLine($"w21: {w21,10:F4}");
        Console.WriteLine($"w22: {w22,10:F4}");
        Console.WriteLine($"b1 : {b1,10:F4}");
        Console.WriteLine($"b2 : {b2,10:F4}");
        Console.WriteLine($"b3 : {b3,10:F4}");

        while (true)
        {
            Console.WriteLine("\nValide esse modelo para porta XOR entrando com os dados seguintes:");
            Console.WriteLine("\n===================================================================");

            Console.Write("x1: ");
            int ix1 = int.Parse(Console.ReadLine());

            Console.Write("x2: ");
            int ix2 = int.Parse(Console.ReadLine());

            // Verifica se os valores de entrada são 0 ou 1
            if ((ix1 != 0 && ix1 != 1) || (ix2 != 0 && ix2 != 1))
            {
                Console.WriteLine("Erro: x1 e x2 devem ser 0 ou 1.");
                continue;
            }

            double iz1 = w11 * ix1 + w21 * ix2 + b1;
            double ih1 = sigmoide(iz1);

            double iz2 = w12 * ix1 + w22 * ix2 + b2;
            double ih2 = sigmoide(iz2);

            double saida = v1 * ih1 + v2 * ih2 + b3;
            Console.WriteLine("");
            Console.WriteLine($"saida: {saida}");
            Console.WriteLine($"sigmoide(saida): {sigmoide(saida)}");
            double result = sigmoide(saida) < 0.5 ? 0 : 1;

            Console.WriteLine("");
            Console.WriteLine($"*************************************************");
            Console.WriteLine($" Resultado final via modelo: x1 -> {ix1} x2 -> {ix2} => {result}");
            Console.WriteLine($"*************************************************");
        }

        return;
    }
}



double sigmoide(double x)
{
    return 1 / (1 + Math.Exp(-x));
}
