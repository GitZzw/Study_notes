## [参考文献1]：(https://docs.ros.org/melodic/api/catkin/html/howto/index.html)
## [参考文献2]：(http://wiki.ros.org/catkin/CMakeLists.txt)
## [参考文献3]：(http://wiki.ros.org/catkin/package.xml)
## [参考文献4]：(https://www.ros.org/reps/rep-0127.html)
## [参考文献5]：(https://www.ros.org/reps/rep-0140.html)

# Catkin编译系统
## 1.catkin概述
    < catkin是ROS的官方编译系统，是ros原始编译系统build的继承者。
      catkin相比于build来讲操作更加简化且工作效率更高，可移植性更好，且支持交叉编译和更加合理的功能包分配。
   
## 2.catkin编译的工作流程
    < 首先在工作空间的src目录下递归的查找每一个ros的package。
      每一个package中都有package.xml和CMakeList.txt文件，catkin依据CMakeList.txt文件，生成makefile文件，放在工作空间的build文件夹中。
      然后make刚刚生成的makefiles等文件，编译链接生成可执行文件，放在工作空间下的devel文件夹中。

## 3.catkin的特点

    < 一个catkin的package必须包括package.xml和CMakeList.txt这两个文件
      package.xml：定义了package的属性（包的自我描述），如包名、版本号、作者、依赖等
      CMakeList.txt：规定了catkin的编译规则，是构建package所需的CMake文件
      具体的有：调用catkin的函数/宏；解析package.xml；找到其他依赖的catkin软件包；将本软件包添加到环境变量等等
 
## 4.指令
    < catkin_make 创建和初始化工作空间，编译
      catkin_create_pkg  创建一个包含package.xml和CMakeList.txt文件的空功能包


# package.xml(功能包配置文件)  参见comments
```
          <?xml version="1.0"?>
          <package format="2">
            <name>test1</name>
            <version>0.0.0</version>
            <description>The test1 package</description>

            <!-- One maintainer tag required, multiple allowed, one person per tag -->
            <!-- Example:  -->
            <!-- <maintainer email="jane.doe@example.com">Jane Doe</maintainer> -->
            <maintainer email="ubuntu@todo.todo">ubuntu</maintainer>


            <!-- One license tag required, multiple allowed, one license per tag -->
            <!-- Commonly used license strings: -->
            <!--   BSD, MIT, Boost Software License, GPLv2, GPLv3, LGPLv2.1, LGPLv3 -->
            <license>TODO</license>


            <!-- Url tags are optional, but multiple are allowed, one per tag -->
            <!-- Optional attribute type can be: website, bugtracker, or repository -->
            <!-- Example: -->
            <!-- <url type="website">http://wiki.ros.org/test1</url> -->


            <!-- Author tags are optional, multiple are allowed, one per tag -->
            <!-- Authors do not have to be maintainers, but could be -->
            <!-- Example: -->
            <!-- <author email="jane.doe@example.com">Jane Doe</author> -->


            <!-- The *depend tags are used to specify dependencies -->
            <!-- Dependencies can be catkin packages or system dependencies -->
            <!-- Examples: -->
            <!-- Use depend as a shortcut for packages that are both build and exec dependencies -->
            <!--   <depend>roscpp</depend> -->
            <!--   Note that this is equivalent to the following: -->
            <!--   <build_depend>roscpp</build_depend> -->
            <!--   <exec_depend>roscpp</exec_depend> -->
            <!-- Use build_depend for packages you need at compile time: -->
            <!--   <build_depend>message_generation</build_depend> -->
            <!-- Use build_export_depend for packages you need in order to build against this package: -->
            <!--   <build_export_depend>message_generation</build_export_depend> -->
            <!-- Use buildtool_depend for build tool packages: -->
            <!--   <buildtool_depend>catkin</buildtool_depend> -->
            <!-- Use exec_depend for packages you need at runtime: -->
            <!--   <exec_depend>message_runtime</exec_depend> -->
            <!-- Use test_depend for packages you need only for testing: -->
            <!--   <test_depend>gtest</test_depend> -->
            <!-- Use doc_depend for packages you need only for building documentation: -->
            <!--   <doc_depend>doxygen</doc_depend> -->
            <buildtool_depend>catkin</buildtool_depend>
            <build_depend>roscpp</build_depend>
            <build_export_depend>roscpp</build_export_depend>
            <exec_depend>roscpp</exec_depend>


            <!-- The export tag contains other, unspecified, tags -->
            <export>
              <!-- Other tools can request additional information be placed here -->

            </export>
          </package>
```


# CMakeList.txt文件,仔细阅读comments
```
            cmake_minimum_required(VERSION 2.8.3)
            project(test1)

            ## Compile as C++11, supported in ROS Kinetic and newer
            # add_compile_options(-std=c++11)

            ## Find catkin macros and libraries
            ## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
            ## is used, also find other catkin packages
            find_package(catkin REQUIRED COMPONENTS
              roscpp
            )

            ## System dependencies are found with CMake's conventions
            # find_package(Boost REQUIRED COMPONENTS system)


            ## Uncomment this if the package has a setup.py. This macro ensures
            ## modules and global scripts declared therein get installed
            ## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
            # catkin_python_setup()

            ################################################
            ## Declare ROS messages, services and actions ##
            ################################################

            ## To declare and build messages, services or actions from within this
            ## package, follow these steps:
            ## * Let MSG_DEP_SET be the set of packages whose message types you use in
            ##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
            ## * In the file package.xml:
            ##   * add a build_depend tag for "message_generation"
            ##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
            ##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
            ##     but can be declared for certainty nonetheless:
            ##     * add a exec_depend tag for "message_runtime"
            ## * In this file (CMakeLists.txt):
            ##   * add "message_generation" and every package in MSG_DEP_SET to
            ##     find_package(catkin REQUIRED COMPONENTS ...)
            ##   * add "message_runtime" and every package in MSG_DEP_SET to
            ##     catkin_package(CATKIN_DEPENDS ...)
            ##   * uncomment the add_*_files sections below as needed
            ##     and list every .msg/.srv/.action file to be processed
            ##   * uncomment the generate_messages entry below
            ##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

            ## Generate messages in the 'msg' folder
            # add_message_files(
            #   FILES
            #   Message1.msg
            #   Message2.msg
            # )

            ## Generate services in the 'srv' folder
            # add_service_files(
            #   FILES
            #   Service1.srv
            #   Service2.srv
            # )

            ## Generate actions in the 'action' folder
            # add_action_files(
            #   FILES
            #   Action1.action
            #   Action2.action
            # )

            ## Generate added messages and services with any dependencies listed here
            # generate_messages(
            #   DEPENDENCIES
            #   std_msgs  # Or other packages containing msgs
            # )

            ################################################
            ## Declare ROS dynamic reconfigure parameters ##
            ################################################

            ## To declare and build dynamic reconfigure parameters within this
            ## package, follow these steps:
            ## * In the file package.xml:
            ##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
            ## * In this file (CMakeLists.txt):
            ##   * add "dynamic_reconfigure" to
            ##     find_package(catkin REQUIRED COMPONENTS ...)
            ##   * uncomment the "generate_dynamic_reconfigure_options" section below
            ##     and list every .cfg file to be processed

            ## Generate dynamic reconfigure parameters in the 'cfg' folder
            # generate_dynamic_reconfigure_options(
            #   cfg/DynReconf1.cfg
            #   cfg/DynReconf2.cfg
            # )

            ###################################
            ## catkin specific configuration ##
            ###################################
            ## The catkin_package macro generates cmake config files for your package
            ## Declare things to be passed to dependent projects
            ## INCLUDE_DIRS: uncomment this if your package contains header files
            ## LIBRARIES: libraries you create in this project that dependent projects also need
            ## CATKIN_DEPENDS: catkin_packages dependent projects also need
            ## DEPENDS: system dependencies of this project that dependent projects also need
            catkin_package(
            #  INCLUDE_DIRS include
            #  LIBRARIES test1
            #  CATKIN_DEPENDS roscpp
            #  DEPENDS system_lib
            )

            ###########
            ## Build ##
            ###########

            ## Specify additional locations of header files
            ## Your package locations should be listed before other locations
            ## include_directories()：添加头文件路径
            include_directories(
            # include
              ${catkin_INCLUDE_DIRS}
            )

            ## Declare a C++ library
            ## add_library()：生成库文件
            # add_library(${PROJECT_NAME}
            #   src/${PROJECT_NAME}/test1.cpp
            # )

            ## Add cmake target dependencies of the library
            ## as an example, code may need to be generated before libraries
            ## either from message generation or dynamic reconfigure
            ## add_dependencies()：添加依赖项，在使用ROS的message、service、action时添加
            # add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

            ## Declare a C++ executable
            ## With catkin_make all packages are built within a single CMake context
            ## The recommended prefix ensures that target names across packages don't collide
            ## add_executable()：生成可执行文件
            # add_executable(${PROJECT_NAME}_node src/test1_node.cpp)

            ## Rename C++ executable without prefix
            ## The above recommended prefix causes long target names, the following renames the
            ## target back to the shorter version for ease of user use
            ## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
            # set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

            ## Add cmake target dependencies of the executable
            ## same as for the library above
            # add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

            ## Specify libraries to link a library or executable target against
            ## 为可执行文件或库添加链接库
            # target_link_libraries(${PROJECT_NAME}_node
            #   ${catkin_LIBRARIES}
            # )

            #############
            ## Install ##
            #############

            # all install targets should use catkin DESTINATION variables
            # See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

            ## Mark executable scripts (Python etc.) for installation
            ## in contrast to setup.py, you can choose the destination
            # install(PROGRAMS
            #   scripts/my_python_script
            #   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
            # )

            ## Mark executables for installation
            ## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
            # install(TARGETS ${PROJECT_NAME}_node
            #   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
            # )

            ## Mark libraries for installation
            ## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
            # install(TARGETS ${PROJECT_NAME}
            #   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            #   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            #   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
            # )

            ## Mark cpp header files for installation
            # install(DIRECTORY include/${PROJECT_NAME}/
            #   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
            #   FILES_MATCHING PATTERN "*.h"
            #   PATTERN ".svn" EXCLUDE
            # )

            ## Mark other files for installation (e.g. launch and bag files, etc.)
            # install(FILES
            #   # myfile1
            #   # myfile2
            #   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
            # )

            #############
            ## Testing ##
            #############

            ## Add gtest based cpp test target and link libraries
            # catkin_add_gtest(${PROJECT_NAME}-test test/test_test1.cpp)
            # if(TARGET ${PROJECT_NAME}-test)
            #   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
            # endif()

            ## Add folders to be run by python nosetests
            # catkin_add_nosetests(test)
```


# 编译静态库和动态库
> 编译库函数的时候，可以选择编译成静态库或者动态库。静态库对应.a文件，动态库对应.so文件(以冒泡排序为例)

## 1.编写StaticBubble.h头文件，路径在软件包的include文件夹下
    ```
        using namespace std;
        void bubbleSort(int array[], int num);
    ```
    
## 2.在软件包的src文件下创建StaticBubble.cpp文件，用于实现上面头文件中定义的函数
     ```
        #include <iostream>
        using namespace std;
        void bubbleSort(int array[], int num){
            for(int i = 0; i < num; i++){
                for(int j = 0; j < num - i - 1; j++){
                    if(array[j] > array[j + 1]){
                        int temp = array[j];
                        array[j] = array[j + 1];
                        array[j + 1] = temp;
                    }
                }
            }
        }
    ```
    
 ## 3.在src文件夹下创建测试程序testBubble.cpp，里面包含main函数
 ```
            #include <iostream>
            #include <StaticBubble.h>

            using namespace std;

            int main(int argc, char **argv){
                int array[5] = {32,54,38,65,98};
                bubbleSort(array, 5);
                for(int i = 0; i < 5; i++){
                    cout << array[i] << '\t';
                }
                cout << endl;

            }
   ```
   
## 4.CMakelist
> add_library中可以选择STATIC或者缺省，代表编译成静态库，即.a文件;选择SHARED时，编译成动态库。
   ```
            cmake_minimum_required(VERSION 3.7)
            project(Static_Lib)

            set(CMAKE_CXX_STANDARD 11)
            #导入头文件和生成链接库
            include_directories(include)
            add_library(StaticBubble STATIC src/StaticBubble.cpp)   #STATIC  or SHARED

            #生成可执行程序，链接库
            add_executable(Static_Lib src/testBubble.cpp)
            target_link_libraries(Static_Lib StaticBubble)
   ```
   
