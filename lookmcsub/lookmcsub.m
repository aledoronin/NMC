% lookmcsub.m
clear all
close all
home

global Nfile J F S A E r z
global mua mus g n1 n2
global mcflag radius waist xs ys zs
global NR NZ dr dz Nphotons


getmcsub('/Users/alexd/Dropbox/NMC/Build/Products/Release/mcOUT0.dat');
%getmcsub('/Users/alexd/Dropbox/NMC/lookmcsub/mcOUT0.dat');
%getmcsub('/Users/alexd/Dropbox/NMC/lookmcsub/mcOUT0.dat');
% Reads the impulse response recorded in mcOUT0.dat
% by mcsub.c, after being called by callmcsub.c.

%% Show fluence
% Plots Fzr(z,rr), a 2D map of fluence rate with rr = -r:+r.
% Overflow bins at rr = 1 and NR, and at z = NZ.

Fzr = zeros(NZ,2*NR);
Fzr(:,NR+[1:NR]) = F;
Fzr(:,[NR:-1:1]) = F;
rr               = zeros(2*NR,1);
rr(NR+[1:NR])    = r;
rr(NR:-1:1)      = -r;


Fzr_1 = Fzr;
J_1 = J;

figure(1);clf
imagesc(rr,z,log10(Fzr_1))
colorbar
set(gca,'fontsize',14)
xlabel('r [cm]')
ylabel('z [cm]')
title('log_{10}(relative fluence rate [W/cm^2 per W delivered])')
colormap('jet')
caxis([-0.5 4.5]);



%getmcsub('/Users/alexd/Dropbox/NMC/Build/Products/Release/mcOUT1or.dat');
getmcsub('/Users/alexd/Dropbox/NMC/lookmcsub/mcOUT0.dat');
Fzr = zeros(NZ,2*NR);
Fzr(:,NR+[1:NR]) = F;
Fzr(:,[NR:-1:1]) = F;
rr               = zeros(2*NR,1);
rr(NR+[1:NR])    = r;
rr(NR:-1:1)      = -r;


Fzr_2 = Fzr;
J_2 = J;

figure(2);clf
imagesc(rr,z,log10(Fzr_2))
colorbar
set(gca,'fontsize',14)
xlabel('r [cm]')
ylabel('z [cm]')
title('log_{10}(relative fluence rate [W/cm^2 per W delivered])')
colormap('jet')
caxis([-0.5 4.5]);

diff = (Fzr_1 - Fzr_2) ./ (Fzr_1 + Fzr_2);
%Dev = abs(Fzr_1-Fzr_2).^2;

figure
imagesc((abs(diff)))
colorbar

figure
semilogy(r, J_1,r, J_2)
legend('neural','original')


