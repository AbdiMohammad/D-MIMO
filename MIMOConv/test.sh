#! /bin/bash

NAME=Mohammad

# for i in 1 2
# do
#     ssh jetson << EOF

#     echo ${NAME}
#     cd ~/MAbdi/D-MIMO/MIMOConv/
#     ls

#     EOF
# done

ssh jetson << EOF
    for i in 1 2
    do
        export NUM=Salam
        echo \${NUM}
        echo ${NAME}
    done
EOF