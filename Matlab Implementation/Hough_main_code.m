% ?????????? ????????????????, ece7888

clear all; clc; close all;

option=1;                 % ??????? ???????

img=imread('/Users/charalamposp/Desktop/1.jpg');
im=im2bw(img,0.2);
im=1-im;
% im=imclose(1-im,strel('disk',10));


theta_step=pi/1024;                          % ???? ?????? ?
th=[-0:theta_step:pi];                     % ????? ?
h = waitbar(0,'Please wait...ece7888');   % ?????????? waitbar
r_limit = norm([size(im,1) size(im,2)]);   
r_range=[-round(r_limit):round(r_limit)]; % ????? r
P=zeros( round(2*r_limit) , numel(th) );  % ???????????? ?????? P

for i=1:size(im,1)                        % ?????? ???????
   for j=1:size(im,2)
      if im(i,j)==1                       % ?? ??????? ??????
          for k=1:length(th)              % ???? ??? ??????????????
            r=i*cos(th(k))+j*sin(th(k));  % ??? ???????? +1 ??? ?(r,?)
            P(round(r+r_limit),k)=P(round(r+r_limit),k)+1; %r+rlimit ????? ?????? ?? ?????? ?????? ??????. ???? ???? ?????????? ??????? ?? ????? r_range.
          end
      end
   end
   waitbar(i/size(im,1));
end   
close(h)        

% ?????????? ?? Hough Space
figure(1)
set(gcf,'numbertitle','off','name','Hough Transform - ????? ???????')
subplot(1,2,1)
imshow(im)
colormap(gray)
title('Input Image')

subplot(1,2,2)
imagesc(th,r_range,P)
colormap(gray)
xlabel('Angle (rad)');
ylabel('R');
title('Hough Transform')
%%
% ??????? ????????? ??? ??? P ???? ?? ????????????? ??? ????????? ???????

threshold=100; 

L=FIND_LINES(size(im,1),size(im,2),threshold,P,th,r,im);

plot([L(:,1),L(:,3)].',[L(:,2),L(:,4)].','-r','linewidth',2)

%%
BL=[];
for i=1:size(L,1) % ??? ???? ??? ??????? ??? L
    bl=LINE_SEGMENTATION(L(i,:),im);
    BL=[BL; bl];
end
BL;



Write_EPS(size(im,2),size(im,1),BL,'Result.eps')

%%

BW = edge(im,'canny');
[H,theta,rho] = hough(BW);
P = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
x = theta(P(:,2));
y = rho(P(:,1));
lines = houghlines(im,theta,rho,P,'FillGap',5,'MinLength',7);


figure, imshow(im), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',5,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',5,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',5,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
% highlight the longest line segment
plot(xy_long(:,1),xy_long(:,2),'LineWidth',5,'Color','red');


