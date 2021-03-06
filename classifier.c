#pragma once
#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "assert.h"
#include "classifier.h"
#include "cuda.h"
#include "image.h"
#ifdef WIN32
#include <time.h>
#include <winsock.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif


#include "darkWrapper.h"
#include "darknet.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/version.hpp"

#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#endif
image get_image_from_stream(CvCapture *cap);
#endif

float *get_regression_values(char **labels, int n)
{
    float *v = calloc(n, sizeof(float));
    int i;
    for(i = 0; i < n; ++i){
        char *p = strchr(labels[i], ' ');
        *p = 0;
        v[i] = atof(p+1);
    }
    return v;
}

void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i;
	 
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network *nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&nets[i], weightfile);
        }
        if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    int imgs = net.batch * net.subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("Number of Imgs %d\n", plist->size);
    int N = plist->size;
    clock_t time;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.threads = 32;
    args.hierarchy = net.hierarchy;

    args.min = net.min_crop;
    args.max = net.max_crop;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.size = net.w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    args.type = CLASSIFICATION_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);
	int epochSave = TRUE;
	int prevEpoch = 0;

    int epoch = (*net.seen)/N;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0)
	{
        time=clock();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        //printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
		printf("Epoch: %d || Batch: %d || Progress: %.3f || loss: %f , %f avg || L.Rate: %f || Loaded in %lf sec. || # of imgs: %d\n",epoch, get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        free_data(train);

        if(epoch>0 && epoch%2 ==0 && epochSave)
		{
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
			epochSave = FALSE;
			prevEpoch = epoch;
        }
		if(prevEpoch!=epoch)
		{
			epochSave = TRUE;
		}
            epoch = *net.seen/N;

  //      if(get_current_batch(net)%100 == 0)
		//{
  //          char buff[256];
  //          sprintf(buff, "%s/%s.backup",backup_directory,base);
  //          save_weights(net, buff);
  //      }
    } //end while()
 

    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}


/*
   void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int clear)
   {
   srand(time(0));
   float avg_loss = -1;
   char *base = basecfg(cfgfile);
   printf("%s\n", base);
   network net = parse_network_cfg(cfgfile);
   if(weightfile){
   load_weights(&net, weightfile);
   }
   if(clear) *net.seen = 0;

   int imgs = net.batch * net.subdivisions;

   printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
   list *options = read_data_cfg(datacfg);

   char *backup_directory = option_find_str(options, "backup", "/backup/");
   char *label_list = option_find_str(options, "labels", "data/labels.list");
   char *train_list = option_find_str(options, "train", "data/train.list");
   int classes = option_find_int(options, "classes", 2);

   char **labels = get_labels(label_list);
   list *plist = get_paths(train_list);
   char **paths = (char **)list_to_array(plist);
   printf("%d\n", plist->size);
   int N = plist->size;
   clock_t time;

   load_args args = {0};
   args.w = net.w;
   args.h = net.h;
   args.threads = 8;

   args.min = net.min_crop;
   args.max = net.max_crop;
   args.angle = net.angle;
   args.aspect = net.aspect;
   args.exposure = net.exposure;
   args.saturation = net.saturation;
   args.hue = net.hue;
   args.size = net.w;
   args.hierarchy = net.hierarchy;

   args.paths = paths;
   args.classes = classes;
   args.n = imgs;
   args.m = N;
   args.labels = labels;
   args.type = CLASSIFICATION_DATA;

   data train;
   data buffer;
   pthread_t load_thread;
   args.d = &buffer;
   load_thread = load_data(args);

   int epoch = (*net.seen)/N;
   while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
   time=clock();

   pthread_join(load_thread, 0);
   train = buffer;
   load_thread = load_data(args);

   printf("Loaded: %lf seconds\n", sec(clock()-time));
   time=clock();

#ifdef OPENCV
if(0){
int u;
for(u = 0; u < imgs; ++u){
    image im = float_to_image(net.w, net.h, 3, train.X.vals[u]);
    show_image(im, "loaded");
    cvWaitKey(0);
}
}
#endif

float loss = train_network(net, train);
free_data(train);

if(avg_loss == -1) avg_loss = loss;
avg_loss = avg_loss*.9 + loss*.1;
printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
if(*net.seen/N > epoch){
    epoch = *net.seen/N;
    char buff[256];
    sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
    save_weights(net, buff);
}
if(get_current_batch(net)%100 == 0){
    char buff[256];
    sprintf(buff, "%s/%s.backup",backup_directory,base);
    save_weights(net, buff);
}
}
char buff[256];
sprintf(buff, "%s/%s.weights", backup_directory, base);
save_weights(net, buff);

free_network(net);
free_ptrs((void**)labels, classes);
free_ptrs((void**)paths, plist->size);
free_list(plist);
free(base);
}
*/

void validate_classifier_crop(char *datacfg, char *filename, char *weightfile)
{
    int i = 0;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;
    float avg_acc = 0;
    float avg_topk = 0;
    int splits = m/1000;
    int num = (i+1)*m/splits - i*m/splits;

    data val, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;

    args.paths = paths;
    args.classes = classes;
    args.n = num;
    args.m = 0;
    args.labels = labels;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i)
	{
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);

        if(i != splits)
		{
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }

        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        float *acc = network_accuracies(net, val, topk);
        avg_acc += acc[0];
        avg_topk += acc[1];
        printf("%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc/i, topk, avg_topk/i, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}

void validate_classifier_10(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        int w = net.w;
        int h = net.h;
        int shift = 32;
        image im = load_image_color(paths[i], w+shift, h+shift);
        image images[10];
        images[0] = crop_image(im, -shift, -shift, w, h);
        images[1] = crop_image(im, shift, -shift, w, h);
        images[2] = crop_image(im, 0, 0, w, h);
        images[3] = crop_image(im, -shift, shift, w, h);
        images[4] = crop_image(im, shift, shift, w, h);
        flip_image(im);
        images[5] = crop_image(im, -shift, -shift, w, h);
        images[6] = crop_image(im, shift, -shift, w, h);
        images[7] = crop_image(im, 0, 0, w, h);
        images[8] = crop_image(im, -shift, shift, w, h);
        images[9] = crop_image(im, shift, shift, w, h);
        float *pred = calloc(classes, sizeof(float));
        for(j = 0; j < 10; ++j){
            float *p = network_predict(net, images[j].data);
            if(net.hierarchy) hierarchy_predictions(p, net.outputs, net.hierarchy, 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            free_image(images[j]);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_classifier_full(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    int size = net.w;
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, size);
        resize_network(&net, resized.w, resized.h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, resized.data);
        if(net.hierarchy) hierarchy_predictions(pred, net.outputs, net.hierarchy, 1);

        free_image(im);
        free_image(resized);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}


void validate_classifier_single(char *datacfg, char *filename, char *weightfile, DarkNet *dark)
{
    int i, j;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net.hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net.w);
        image crop = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, crop.data);
        if(net.hierarchy) hierarchy_predictions(pred, net.outputs, net.hierarchy, 1);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
			if (indexes[j] == class)
			{
				avg_topk += 1;
				j = topk;
			}
        }
		
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_classifier_multi(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    int scales[] = {224, 288, 320, 352, 384};
    int nscales = sizeof(scales)/sizeof(scales[0]);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        float *pred = calloc(classes, sizeof(float));
        image im = load_image_color(paths[i], 0, 0);
        for(j = 0; j < nscales; ++j){
            image r = resize_min(im, scales[j]);
            resize_network(&net, r.w, r.h);
            float *p = network_predict(net, r.data);
            if(net.hierarchy) hierarchy_predictions(p, net.outputs, net.hierarchy, 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            flip_image(r);
            p = network_predict(net, r.data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            if(r.data != im.data) free_image(r);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void try_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int layer_num)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image orig = load_image_color(input, 0, 0);
        image r = resize_min(orig, 256);
        image im = crop_image(r, (r.w - 224 - 1)/2 + 1, (r.h - 224 - 1)/2 + 1, 224, 224);
        float mean[] = {0.48263312050943, 0.45230225481413, 0.40099074308742};
        float std[] = {0.22590347483426, 0.22120921437787, 0.22103996251583};
        float var[3];
        var[0] = std[0]*std[0];
        var[1] = std[1]*std[1];
        var[2] = std[2]*std[2];

        normalize_cpu(im.data, mean, var, 1, 3, im.w*im.h);

        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);

        layer l = net.layers[layer_num];
        for(i = 0; i < l.c; ++i){
            if(l.rolling_mean) printf("%f %f %f\n", l.rolling_mean[i], l.rolling_variance[i], l.scales[i]);
        }
#ifdef GPU
        cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
        for(i = 0; i < l.outputs; ++i){
            printf("%f\n", l.output[i]);
        }
        /*

           printf("\n\nWeights\n");
           for(i = 0; i < l.n*l.size*l.size*l.c; ++i){
           printf("%f\n", l.filters[i]);
           }

           printf("\n\nBiases\n");
           for(i = 0; i < l.n; ++i){
           printf("%f\n", l.biases[i]);
           }
         */

        top_predictions(net, top, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
        if (filename) break;
    }
}

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    int size = net.w;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
		image r = letterbox_image(im, net.w, net.h);
        //image r = resize_min(im, size);
        //resize_network(&net, r.w, r.h);
        printf("%d %d\n", r.w, r.h);

        float *X = r.data;
        time=clock();
        float *predictions = network_predict(net, X);
        if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
        top_k(predictions, net.outputs, top, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
            else printf("%s: %f\n",names[index], predictions[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        if (filename) break;
    }
}


void label_classifier(char *datacfg, char *filename, char *weightfile)
{
    int i;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "names", "data/labels.list");
    char *test_list = option_find_str(options, "test", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net.w);
        image crop = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        float *pred = network_predict(net, crop.data);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        int ind = max_index(pred, classes);

        printf("%s\n", labels[ind]);
    }
}
//
//#ifdef MAKE_DLL
//network Parse_Network_CFG(char* cfgfile)
//{
//	return parse_network_cfg(cfgfile);
//}
//#endif


#ifdef MAKE_DLL
void test_classifier(DarkNet *dark)
{
	int curr = 0;
	char datafile[100];
	char cfgfile[100];
	char weightfile[100];
	char label[100];
	sprintf(datafile, "C:/WISVision/ADC_Model/%s/%s.data", dark->m_strModelName, dark->m_strModelName);
	sprintf(cfgfile, "C:/WISVision/ADC_Model/%s/%s.cfg", dark->m_strModelName, dark->m_strModelName);
	sprintf(weightfile, "C:/WISVision/ADC_Model/%s/%s.weights", dark->m_strModelName, dark->m_strModelName);
	sprintf(label, "C:/WISVision/ADC_Model/%s/%s.list", dark->m_strModelName, dark->m_strModelName);

	if (!dark->m_bConfigLoaded)
	{
		FILE *file = fopen(cfgfile, "r");
		if (file == 0) return CFG_LOAD_FAIL;
		else   fclose(file);
		dark->net = parse_network_cfg(cfgfile); //네트워크 불러오기
		dark->m_bConfigLoaded = true;
	}

	if (!dark->m_bWeightLoaded && weightfile)
	{
		FILE *file = fopen(weightfile, "r");
		if (file == 0) return WEIGHT_LOAD_FAIL;
		else   fclose(file);
		load_weights(&dark->net, weightfile);
		dark->m_bWeightLoaded = true;
	}
 
	FILE *file = fopen(datafile, "r");
	if (file == 0) return DATA_LOAD_FAIL;
	else   fclose(file);
	list *options = read_data_cfg(datafile);
	*dark->m_pClassNum = option_find_int(options, "classes", 2);

	data val, buffer;
	load_args args = { 0 };
	args.w = dark->net.w;
	args.h = dark->net.h;
	args.paths = NULL; //  (char **)list_to_array(plist);
	args.classes = *(dark->m_pClassNum);
	args.n = dark->net.batch;
	args.m = 0;
	args.labels = 0;
	args.d = &buffer;
	args.type = OLD_CLASSIFICATION_DATA;

	pthread_t load_thread = load_data_in_thread(args,dark);

	srand(time(0));
	clock_t time;
	time = clock();

	for (curr = 0; curr < dark->m_nTotalDefectImage; curr += dark->net.batch)
	{
		// allocate image buffer 
		float **imgBatch = (float**)malloc(sizeof(float*) * dark->net.batch);
		for (int i = 0; i < dark->net.batch; i++)
			imgBatch[i] = (float*)malloc(sizeof(float) * dark->net.w * dark->net.h * dark->net.c);

		pthread_join(load_thread, 0);
		val = buffer;
		
		if (curr < dark->m_nTotalDefectImage)
		{
			if (curr + dark->net.batch > dark->m_nTotalDefectImage) args.n = dark->m_nTotalDefectImage - curr;
			load_thread = load_data_in_thread(args);
		}
		//fprintf(stderr, "Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock() - time));

		//Resize Image Data to Fit Current Network
		extractImageBatch(dark->m_ppDefectImg, dark->m_nDefectImgWidth, dark->m_nDefectImgHeight, imgBatch, curr, dark->net.batch, dark->net.w, true);

		val.X.rows = args.n;
		val.X.cols = dark->net.w*dark->net.h*dark->net.c;
		val.X.vals = calloc(val.X.rows, sizeof(float*));
		val.X.vals = imgBatch;

		// Image Classification
		matrix pred = network_predict_data(dark->net, val);

		// Save Prediction Result to send back to VisionWorks
		for (int i = 0; i < pred.rows; ++i)
		{
			for (int j = 0; j < pred.cols; ++j)
			{
				dark->predictionResult[curr+i][j]= pred.vals[i][j];
			}
		}

		// Free allocated memory
		free_matrix(pred);
		free_data(val);
	}
		//fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock() - time), val.X.rows, curr - dark->net.batch + val.X.rows);
	
	// free GPU memory
	free_network(dark->net); 
	return SUCCESS;
}



int compare(const void * a, const void * b)
{
	return (*(int*)a - *(int*)b);
}

void test_classifier_valid(DarkNet *dark)
{
	int curr = 0;
	char datafile[100];
	char cfgfile[100];
	char weightfile[100];
	char label[100];
	sprintf(datafile, "C:/WISVision/ADC_Model/%s/%s.data", dark->m_strModelName, dark->m_strModelName);
	sprintf(cfgfile, "C:/WISVision/ADC_Model/%s/%s.cfg", dark->m_strModelName, dark->m_strModelName);
	sprintf(weightfile, "C:/WISVision/ADC_Model/%s/%s.weights", dark->m_strModelName, dark->m_strModelName);
	sprintf(label, "C:/WISVision/ADC_Model/%s/%s.list", dark->m_strModelName, dark->m_strModelName);

	if (!dark->m_bConfigLoaded)
	{
		FILE *file = fopen(cfgfile, "r");
		if (file == 0) return CFG_LOAD_FAIL;
		else   fclose(file);
		dark->net = parse_network_cfg(cfgfile); //네트워크 불러오기
		dark->m_bConfigLoaded = true;
	}

	if (!dark->m_bWeightLoaded && weightfile)
	{
		FILE *file = fopen(weightfile, "r");
		if (file == 0) return WEIGHT_LOAD_FAIL;
		else   fclose(file);
		load_weights(&dark->net, weightfile);
		dark->m_bWeightLoaded = true;
	}

	FILE *file = fopen(datafile, "r");
	if (file == 0) return DATA_LOAD_FAIL;
	else   fclose(file);
	list *options = read_data_cfg(datafile);
	*dark->m_pClassNum = option_find_int(options, "classes", 2);

	data val, buffer;
	load_args args = { 0 };
	args.w = dark->net.w;
	args.h = dark->net.h;
	args.paths = NULL; //  (char **)list_to_array(plist);
	args.classes = *(dark->m_pClassNum);
	args.n = dark->net.batch;
	args.m = 0;
	args.labels = 0;
	args.d = &buffer;
	args.type = OLD_CLASSIFICATION_DATA;

	pthread_t load_thread = load_data_in_thread(args, dark);

	srand(time(0));
	clock_t time;
	time = clock();
	IplImage *img = cvCreateImage(cvSize(args.w, args.h), IPL_DEPTH_8U, 3);

	char **labels = get_labels(label);

	for (curr = 0; curr < dark->m_nTotalDefectImage; curr += dark->net.batch)
	{
		// allocate image buffer 
		float **imgBatch = (float**)malloc(sizeof(float*) * dark->net.batch);
		for (int i = 0; i < dark->net.batch; i++)
			imgBatch[i] = (float*)malloc(sizeof(float) * dark->net.w * dark->net.h * dark->net.c);

		pthread_join(load_thread, 0);
		val = buffer;

		if (curr < dark->m_nTotalDefectImage)
		{
			if (curr + dark->net.batch > dark->m_nTotalDefectImage) args.n = dark->m_nTotalDefectImage - curr;
			load_thread = load_data_in_thread(args);
		}
		//fprintf(stderr, "Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock() - time));

		//Resize Image Data to Fit Current Network
		extractImageBatch(dark->m_ppDefectImg, dark->m_nDefectImgWidth, dark->m_nDefectImgHeight, imgBatch, curr, dark->net.batch, dark->net.w, true);

		val.X.rows = args.n;
		val.X.cols = dark->net.w*dark->net.h*dark->net.c;
		val.X.vals = calloc(val.X.rows, sizeof(float*));
		val.X.vals = imgBatch;

		// Image Classification
		matrix pred = network_predict_data(dark->net, val);

		// Save Prediction Result to send back to VisionWorks
		printf("\nLabel_   : %s  %s  %s  %s  %s   %s  %s ", labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6]);
		for (int i = 0; i < pred.rows; ++i)
		{
			float max = 0;
			int maxLabel = -1;
			printf("\nLabel_#.%d :", curr + i);
			for (int j = 0; j < pred.cols; ++j)
			{
				for (int m = 20; m < args.h; m++)
				{
					for (int n = 0; n < args.w; n++)
					{
						img->imageData[m*args.w*3 + n*3  ] = (unsigned char)(val.X.vals[i][(m)*(args.w) + n ]*255);
						img->imageData[m*args.w*3 + n*3+1] = (unsigned char)(val.X.vals[i][(m)*(args.w) + n + (args.w)*(args.h)] * 255);
						img->imageData[m*args.w*3 + n*3+2] = (unsigned char)(val.X.vals[i][(m)*(args.w) + n + (args.w)*(args.h)*2] * 255);

					}
				}
				for (int m = 0; m < 20; m++)
				{
					for (int n = 0; n < args.w; n++)
					{
						img->imageData[m*args.w * 3 + n * 3]     =0;
						img->imageData[m*args.w * 3 + n * 3 + 1] =0;
						img->imageData[m*args.w * 3 + n * 3 + 2] =0;

					}
				}

				dark->predictionResult[curr + i][j] = pred.vals[i][j];
				printf("%0.3f   ",pred.vals[i][j]);
			
				if (max < dark->predictionResult[curr + i][j])
				{
					max = dark->predictionResult[curr + i][j];
					maxLabel = j;
				}
			}
			char text[255];
			sprintf(text, "%s (%0.3f)", labels[maxLabel], max);
			CvFont font;
			cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX , 0.5, 0.5,0,0.8,0);
			cvPutText(img, text, cvPoint(10, 13), &font, cvScalar(0, 255, 0,0));

			cvShowImage("img", img);
			char c = cvWaitKey(0);
			if(c == 'a')	i -= 2;
			else if (c == '1')	i += 10;
			else if (c == 'n')	i += pred.rows;
			else if (c == 's')
			{
				char name[100];
				int p[3] = { CV_IMWRITE_JPEG_QUALITY,100,0 };
				sprintf(name, "C:/WISVision/ADC_Model/%s/%d_%s(%0.3f).jpg", dark->m_strModelName, curr+i,labels[maxLabel],max);
				cvSaveImage(name, img, p);
			}
			else if (c == 27) return true;
			if (i < 0)i = 0;
		}

		// Free allocated memory
		free_matrix(pred);
		free_data(val);
	}
	//fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock() - time), val.X.rows, curr - dark->net.batch + val.X.rows);

	// free GPU memory
	free_network(dark->net);
	return SUCCESS;
}



int get_classified_image(DarkNet *dark) 
{
	SYSTEMTIME st;
	GetSystemTime(&st);
	int curr = 0;
	char datafile[100];
	char cfgfile[100];
	char weightfile[100];
	char label[100];
	sprintf(datafile, "C:/WISVision/ADC_Model/%s/%s.data", dark->m_strModelName, dark->m_strModelName);
	sprintf(cfgfile, "C:/WISVision/ADC_Model/%s/%s.cfg", dark->m_strModelName, dark->m_strModelName);
	sprintf(weightfile, "C:/WISVision/ADC_Model/%s/%s.weights", dark->m_strModelName, dark->m_strModelName);
	sprintf(label, "C:/WISVision/ADC_Model/%s/%s.list", dark->m_strModelName, dark->m_strModelName);


	char TimeDir[100];
	char rcpDir[100];
	CreateDirectory("C:/WISVision/ADC_Model/savedIMG/", NULL);
	sprintf(TimeDir, "C:/WISVision/ADC_Model/savedIMG/%d_%d_%d", st.wYear, st.wMonth, st.wDay);
	sprintf(rcpDir, "%s/%d_%d", TimeDir,st.wHour,st.wMinute);

	list *options = read_data_cfg(datafile);
	*dark->m_pClassNum = option_find_int(options, "classes", 2);

	char **labels = get_labels(label);

	CreateDirectory(TimeDir, NULL);
	CreateDirectory(rcpDir, NULL);

	char** labelPath = (char**)malloc(sizeof(char*) * (*dark->m_pClassNum));
	for (int i = 0; i < (*dark->m_pClassNum); i++)
	{
		labelPath[i] = (char*)malloc(sizeof(char) * 100);
	}
	for (int i = 0; i < (*dark->m_pClassNum); i++)
	{
		sprintf(labelPath[i], "%s/%d_%s", rcpDir, i, labels[i]);
		CreateDirectory(labelPath[i], NULL);
	}

	if (!dark->m_bConfigLoaded)
	{
		FILE *file = fopen(cfgfile, "r");
		if (file == 0) return CFG_LOAD_FAIL;
		else   fclose(file);
		dark->net = parse_network_cfg(cfgfile); //네트워크 불러오기
		dark->m_bConfigLoaded = true;
	}

	if (!dark->m_bWeightLoaded && weightfile)
	{
		FILE *file = fopen(weightfile, "r");
		if (file == 0) return WEIGHT_LOAD_FAIL;
		else   fclose(file);
		load_weights(&dark->net, weightfile);
		dark->m_bWeightLoaded = true;
	}

	FILE *file = fopen(datafile, "r");
	if (file == 0) return DATA_LOAD_FAIL;
	else   fclose(file);

	data val, buffer;
	load_args args = { 0 };
	args.w = dark->net.w;
	args.h = dark->net.h;
	args.paths = NULL; //  (char **)list_to_array(plist);
	args.classes = *(dark->m_pClassNum);
	args.n = dark->net.batch;
	args.m = 0;
	args.labels = 0;
	args.d = &buffer;
	args.type = OLD_CLASSIFICATION_DATA;

	pthread_t load_thread = load_data_in_thread(args, dark);

	srand(time(0));
	clock_t time;
	time = clock();
	IplImage *img = cvCreateImage(cvSize(args.w, args.h), IPL_DEPTH_8U, 3);


	for (curr = 0; curr < dark->m_nTotalDefectImage; curr += dark->net.batch)
	{
		// allocate image buffer 
		float **imgBatch = (float**)malloc(sizeof(float*) * dark->net.batch);
		for (int i = 0; i < dark->net.batch; i++)
			imgBatch[i] = (float*)malloc(sizeof(float) * dark->net.w * dark->net.h * dark->net.c);

		pthread_join(load_thread, 0);
		val = buffer;

		if (curr < dark->m_nTotalDefectImage)
		{
			if (curr + dark->net.batch > dark->m_nTotalDefectImage) args.n = dark->m_nTotalDefectImage - curr;
			load_thread = load_data_in_thread(args);
		}
		//fprintf(stderr, "Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock() - time));

		//Resize Image Data to Fit Current Network
		extractImageBatch(dark->m_ppDefectImg, dark->m_nDefectImgWidth, dark->m_nDefectImgHeight, imgBatch, curr, dark->net.batch, dark->net.w, true);

		val.X.rows = args.n;
		val.X.cols = dark->net.w*dark->net.h*dark->net.c;
		val.X.vals = calloc(val.X.rows, sizeof(float*));
		val.X.vals = imgBatch;

		// Image Classification
		matrix pred = network_predict_data(dark->net, val);

		// Save Prediction Result to send back to VisionWorks
		printf("\nLabel_   : %s  %s  %s  %s  %s   %s  %s ", labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6]);
		for (int i = 0; i < pred.rows; ++i)
		{
			float max = 0;
			int maxLabel = -1;
			printf("\nLabel_#.%d :", curr + i);
			for (int j = 0; j < pred.cols; ++j)
			{
				for (int m = 0; m < args.h; m++)
				{
					for (int n = 0; n < args.w; n++)
					{
						img->imageData[m*args.w * 3 + n * 3] = (unsigned char)(val.X.vals[i][(m)*(args.w) + n] * 255);
						img->imageData[m*args.w * 3 + n * 3 + 1] = (unsigned char)(val.X.vals[i][(m)*(args.w) + n + (args.w)*(args.h)] * 255);
						img->imageData[m*args.w * 3 + n * 3 + 2] = (unsigned char)(val.X.vals[i][(m)*(args.w) + n + (args.w)*(args.h) * 2] * 255);
					}
				}

				dark->predictionResult[curr + i][j] = pred.vals[i][j];
				printf("%0.3f   ", pred.vals[i][j]);

				if (max < dark->predictionResult[curr + i][j])
				{
					max = dark->predictionResult[curr + i][j];
					maxLabel = j;
				}
			}

			char name[100];
			int p[3] = { CV_IMWRITE_JPEG_QUALITY,100,0 };
			sprintf(name, "%s/%s(%0.3f)_%d.jpg", labelPath[maxLabel],  labels[maxLabel], max, curr + i);
			cvSaveImage(name, img, p);
		}

		// Free allocated memory
		free_matrix(pred);
		free_data(val);
	}
	//fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock() - time), val.X.rows, curr - dark->net.batch + val.X.rows);

	// free GPU memory
	free_network(dark->net);
	for (size_t i = 0; i < (*dark->m_pClassNum); i++)
	{
		free(labelPath[i]);
	}
	free(labelPath);
	return SUCCESS;
}
#else


int test_classifier(char *datacfg, char *cfgfile, char *weightfile, DarkNet *dark)
{
    int curr = 0;
	network net;
	DarkError darkerror;
	if (!dark->m_bConfigLoaded)
	{
		FILE *file = fopen(cfgfile, "r");
		if (file == 0) return CFG_LOAD_FAIL;
		else   fclose(file);
		net = parse_network_cfg(cfgfile); //네트워크 불러오기
		dark->m_bConfigLoaded = true;
	}
 
    if(!dark->m_bWeightLoaded && weightfile)
	{
		FILE *file = fopen(weightfile, "r");
		if (file == 0) return WEIGHT_LOAD_FAIL;
		else   fclose(file);
        load_weights(&net, weightfile);
		dark->m_bWeightLoaded = true;
	}

    srand(time(0));
 
	char *test_list = "C:/Users/ati/Documents/ChanheeJean/TensorPy_TFLearn/DB/CopiedIMG/tt.list";

    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size; // 전체 이미지 몇장인지
    free_list(plist);    

	list *options = read_data_cfg(datacfg);
	int classes= option_find_int(options, "classes", 2);
	net.batch = 128;
	dark->net.batch = 128;
	dark->net.subdivisions = 1;
	net.subdivisions=1;
    clock_t time;

    data val, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.classes = classes;
    args.n = net.batch;
    args.m = 0;
    args.labels = 0;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
	m = 702;

    for(curr = 0; curr < m+net.batch-1; curr += net.batch)
	{
		float **imgBatch = (float**)malloc(sizeof(float*) * dark->net.batch);
		for (int i = 0; i < dark->net.batch; i++)
			imgBatch[i] = (float*)malloc(sizeof(float) * dark->net.w * dark->net.h * dark->net.c);

        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;
		imgBatch = val.X.vals;

        if(curr < m)
		{
            args.paths = paths + curr;
            if (curr + net.batch > m) args.n = m - curr;
            load_thread = load_data_in_thread(args);
        }
        fprintf(stderr, "Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

	 

	    time=clock();                                   
		val.X.rows = args.n;
        matrix pred = network_predict_data(net, val);
 

		int i, j;
        for(i = 0; i < pred.rows; ++i)
		{
			printf("   || prediction #.%d :  ",i + curr);
            for(j = 0; j < pred.cols; ++j){
                printf("   %0.3f", pred.vals[i][j]);
            }
            printf("\n");
        }

        free_matrix(pred);

        fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock()-time), val.X.rows, curr+dark->net.batch);
        free_data(val);
    }
	printf("Clear CUDA MEMORY?? \n");
	system("pause");
	free_network(net);
	system("pause");
	return true;
}
#endif

void threat_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    float threat = 0;
    float roll = .2;

    printf("Classifier Demo\n");
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    CvCapture * cap;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    int top = option_find_int(options, "top", 1);

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    //cvNamedWindow("Threat", CV_WINDOW_NORMAL); 
    //cvResizeWindow("Threat", 512, 512);
    float fps = 0;
    int i;

    int count = 0;

    while(1){
        ++count;
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        if(!in.data) break;
        image in_s = resize_image(in, net.w, net.h);

        image out = in;
        int x1 = out.w / 20;
        int y1 = out.h / 20;
        int x2 = 2*x1;
        int y2 = out.h - out.h/20;

        int border = .01*out.h;
        int h = y2 - y1 - 2*border;
        int w = x2 - x1 - 2*border;

        float *predictions = network_predict(net, in_s.data);
        float curr_threat = 0;
        if(1){
            curr_threat = predictions[0] * 0 + 
                predictions[1] * .6 + 
                predictions[2];
        } else {
            curr_threat = predictions[218] +
                predictions[539] + 
                predictions[540] + 
                predictions[368] + 
                predictions[369] + 
                predictions[370];
        }
        threat = roll * curr_threat + (1-roll) * threat;

        draw_box_width(out, x2 + border, y1 + .02*h, x2 + .5 * w, y1 + .02*h + border, border, 0,0,0);
        if(threat > .97) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .02*h - 2*border, 
                    x2 + .5 * w + 6*border, 
                    y1 + .02*h + 3*border, 3*border, 1,0,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .02*h - 2*border, 
                x2 + .5 * w + 6*border, 
                y1 + .02*h + 3*border, .5*border, 0,0,0);
        draw_box_width(out, x2 + border, y1 + .42*h, x2 + .5 * w, y1 + .42*h + border, border, 0,0,0);
        if(threat > .57) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .42*h - 2*border, 
                    x2 + .5 * w + 6*border, 
                    y1 + .42*h + 3*border, 3*border, 1,1,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .42*h - 2*border, 
                x2 + .5 * w + 6*border, 
                y1 + .42*h + 3*border, .5*border, 0,0,0);

        draw_box_width(out, x1, y1, x2, y2, border, 0,0,0);
        for(i = 0; i < threat * h ; ++i){
            float ratio = (float) i / h;
            float r = (ratio < .5) ? (2*(ratio)) : 1;
            float g = (ratio < .5) ? 1 : 1 - 2*(ratio - .5);
            draw_box_width(out, x1 + border, y2 - border - i, x2 - border, y2 - border - i, 1, r, g, 0);
        }
        top_predictions(net, top, indexes);
        char buff[256];
        sprintf(buff, "/home/pjreddie/tmp/threat_%06d", count);
        //save_image(out, buff);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);
        }

        if(1){
            show_image(out, "Threat");
            cvWaitKey(10);
        }
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}


void gun_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    int bad_cats[] = {218, 539, 540, 1213, 1501, 1742, 1911, 2415, 4348, 19223, 368, 369, 370, 1133, 1200, 1306, 2122, 2301, 2537, 2823, 3179, 3596, 3639, 4489, 5107, 5140, 5289, 6240, 6631, 6762, 7048, 7171, 7969, 7984, 7989, 8824, 8927, 9915, 10270, 10448, 13401, 15205, 18358, 18894, 18895, 19249, 19697};

    printf("Classifier Demo\n");
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    CvCapture * cap;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    int top = option_find_int(options, "top", 1);

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow("Threat Detection", CV_WINDOW_NORMAL); 
    cvResizeWindow("Threat Detection", 512, 512);
    float fps = 0;
    int i;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        image in_s = resize_image(in, net.w, net.h);
        show_image(in, "Threat Detection");

        float *predictions = network_predict(net, in_s.data);
        top_predictions(net, top, indexes);

        printf("\033[2J");
        printf("\033[1;1H");

        int threat = 0;
        for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i){
            int index = bad_cats[i];
            if(predictions[index] > .01){
                printf("Threat Detected!\n");
                threat = 1;
                break;
            }
        }
        if(!threat) printf("Scanning...\n");
        for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i){
            int index = bad_cats[i];
            if(predictions[index] > .01){
                printf("%s\n", names[index]);
            }
        }

        free_image(in_s);
        free_image(in);

        cvWaitKey(10);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}

void demo_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    printf("Classifier Demo\n");
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    CvCapture * cap;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    int top = option_find_int(options, "top", 1);

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow("Classifier", CV_WINDOW_NORMAL); 
    cvResizeWindow("Classifier", 512, 512);
    float fps = 0;
    int i;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        image in_s = resize_image(in, net.w, net.h);
        show_image(in, "Classifier");

        float *predictions = network_predict(net, in_s.data);
        if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 1);
        top_predictions(net, top, indexes);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);
        }

        free_image(in_s);
        free_image(in);

        cvWaitKey(10);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}

//#ifndef MAKE_DLL
int run_classifier(int argc, char **argv, DarkNet *dark)
{
	DarkError darkerror;
	if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;

    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int top = find_int_arg(argc, argv, "-t", 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    char *layer_s = (argc > 7) ? argv[7]: 0;
    int layer = layer_s ? atoi(layer_s) : -1;
    if(0==strcmp(argv[2], "predict")) predict_classifier(data, cfg, weights, filename, top);
    else if(0==strcmp(argv[2], "try")) try_classifier(data, cfg, weights, filename, atoi(layer_s));

    else if(0==strcmp(argv[2], "train")) train_classifier(data, cfg, weights, gpus, ngpus, clear);

    else if(0==strcmp(argv[2], "demo")) demo_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "gun")) gun_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "threat")) threat_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "label")) label_classifier(data, cfg, weights);

#ifdef MAKE_DLL
	else if (0 == strcmp(argv[2], "test")) test_classifier(dark);
#else
    else if(0==strcmp(argv[2], "test")) darkerror= test_classifier(data, cfg, weights, dark);
#endif
    else if(0==strcmp(argv[2], "valid")) validate_classifier_single(data, cfg, weights,dark);

    else if(0==strcmp(argv[2], "validmulti")) validate_classifier_multi(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid10")) validate_classifier_10(data, cfg, weights);
    else if(0==strcmp(argv[2], "validcrop")) validate_classifier_crop(data, cfg, weights);
    else if(0==strcmp(argv[2], "validfull")) validate_classifier_full(data, cfg, weights);
	return darkerror;
}
 
 