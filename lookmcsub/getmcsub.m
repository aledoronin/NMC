function getmcsub(filename)

global Nfile J F S A E r z
global mua mus g n1 n2
global mcflag radius waist xs ys zs
global NR NZ dr dz Nphotons PLOTON PRINTON

% SaveFile(	Nfile, J, F, S, A, E,       /* save to "mcOUTi.dat", i = Nfile */
% 			mua, mus, g, n1, n2, 
% 		 	mcflag, radius, waist, xs, ys, zs, 
% 		 	NR, NZ, dr, dz, Nphotons);  

fid = fopen(filename,'r');

s = fgetl(fid);
mua = sscanf(s,'%f');
s = fgetl(fid);
mus = sscanf(s,'%f');
s = fgetl(fid);
g = sscanf(s,'%f');
s = fgetl(fid);
n1 = sscanf(s,'%f');
s = fgetl(fid);
n2 = sscanf(s,'%f');

s = fgetl(fid);
mcflag = sscanf(s,'%f');
s = fgetl(fid);
radius = sscanf(s,'%f');
s = fgetl(fid);
waist = sscanf(s,'%f');
s = fgetl(fid);
xs = sscanf(s,'%f');
s = fgetl(fid);
ys = sscanf(s,'%f');
s = fgetl(fid);
zs = sscanf(s,'%f');

s = fgetl(fid);
NR = sscanf(s,'%f');
s = fgetl(fid);
NZ = sscanf(s,'%f');
s = fgetl(fid);
dr = sscanf(s,'%f');
s = fgetl(fid);
dz = sscanf(s,'%f');
s = fgetl(fid);
Nphotons = sscanf(s,'%f');

s = fgetl(fid);
S = sscanf(s,'%f');
s = fgetl(fid);
A = sscanf(s,'%f');
s = fgetl(fid);
E = sscanf(s,'%f');

U = fscanf(fid,'%f',[1 NR+1]);
r = U(2:NR+1)';
U = fscanf(fid,'%f',[1 NR+1]);
J = U(2:NR+1)';
for i=1:NZ
    U = fscanf(fid,'%f', [1 NR+1]);
    %disp(sprintf('z = U(1) = %0.4e', U(1)))
    z(i,1) = U(1);  
    F(i,1:NR) = U(2:NR+1);
end

% disp(sprintf('mua = %0.4f cm^-1', mua))
% disp(sprintf('mus = %0.4f cm^-1', mus))
% disp(sprintf('g   = %0.4f ', g))

fclose(fid);
