clc;clear all;close all
N=8;
X = linspace(0,pi*3,1000);
Y = bsxfun(@(x,n)sin(x+2*n*pi/N), X.', 1:N);
C = linspecer(N);

figure;
%%
files = dir('plots/*.mat');
newcolors = colorcube(8);
set(gca,'colororder',C,'NextPlot', 'replacechildren');
for i = 1:size(files)
    load(sprintf("plots/%s",files(i).name));
    name = strsplit(files(i).name, '_');
    n = name(1);n = n{1};
    
    lr = name(2); lr= lr{1};
    m = name(3); m = m{1};
    bs = name(4); bs = strsplit(bs{1}, '.mat'); bs = bs(1); bs = bs{1};
    plot(TrainLoss,'LineWidth',4,"DisplayName", sprintf("%s, bs=%s, \\alpha=%s, \\mu=%s",n,bs,lr,m));
    hold on;
end

%%
ylim([0, 1]);
xlim([0,200]);
legend('NumColumns',1,'Location','northeast');
grid on;
xlabel('Epoch'); ylabel('Loss'); title('Train Loss - MNIST - LeNet');
set(gca,'YScale', 'log');
set(gca,'FontSize',24);
set(gcf, 'Position',  [620,526,620,452])
print('lenet_train_loss','-depsc');
