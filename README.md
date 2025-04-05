Resolvendo a Porta XOR com Backpropagation
==========================================

Algoritmo original desenvolvido em C# .NET Core 8.0 – Console Application – baseado no vídeo do canal Global Science Network.

Estrutura

Rede com uma camada interna de dois neurônios - entrada x1,  x1 e viés (b1,b2,b3).

Pesos iniciais:

const int epocas = 100000;

const double n = 0.1; // taxa aprendizado

double w11 = 0.50;

double w21 = 0.30;

double b1 = 0.20;

double w12 = 0.40;

double w22 = 0.60;

double b2 = 0.10;

double v1 = 0.30;

double v2 = 0.40;

double b3 = 1;

RESULTADO
=========

***Porta XOR - Backpropagation - Erro aceitável (< ou = a 0,02) encontrado na época: 11721/100000

Amostra 0 = x1 -> 0 x2 -> 0  Real(y) -> 0  Predito(yy) -> 0,1017 Erro -> 0,0052

Amostra 1 = x1 -> 0 x2 -> 1  Real(y) -> 1  Predito(yy) -> 0,8593 Erro -> 0,0099

Amostra 2 = x1 -> 1 x2 -> 0  Real(y) -> 1  Predito(yy) -> 0,8591 Erro -> 0,0099

Amostra 3 = x1 -> 1 x2 -> 1  Real(y) -> 0  Predito(yy) -> 0,1999 Erro -> 0,0200

Pesos resultantes do modelo após 11721 epocas:

v1 :    -6,3896

v2 :     5,5728

w11:    -5,4886

w12:    -2,8264

w21:    -5,5266

w21:    -5,5266

w22:    -2,8210

b1 :     1,6699

b2 :     3,9548

b3 :    -2,2695

Bibliografia
============

Global Science Network. Global Science Network. Redes Neurais Explicadas: Resolvendo a Porta Lógica XOR com Retro-programação. Disponível em https://youtu.be/xeSqeMPMb-0?si=S7QODVY9MKMjwth- Acesso em 2025-04-05 11:21 hh:mm
