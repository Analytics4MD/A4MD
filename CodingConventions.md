# Source code formatting conventions 

(Borrowed from https://bitbucket.org/glotzer/hoomd-blue)

[TOC]

The following conventions apply to all code written in a4md, whether it be C++, python, CMake, etc. Pull requests
will not be accepted and feature branches will not be merged unless these conventions are followed.

## Indentation

* *Spaces* are used to indent lines. (see editor configuration guide below)
* A single level of indentation is to be *4 spaces*
* There should be no whitespace at the end of lines in the file.
* The indentation style used is [Whitesmith's style](http://en.wikipedia.org/wiki/Indent_style#Whitesmiths_style). An extended set of examples follows:

```
#!c++
class SomeClass
    {
    public:
        SomeClass();
        int SomeMethod(int a);
    private:
        int m_some_member;
    };

// indent function bodies
int SomeClass::SomeMethod(int a)
    {
    // indent loop bodies
    while (condition)
        {
        b = a + 1;
        c = b - 2;
        }

    // indent switch bodies and the statements inside each case
    switch (b)
        {
        case 0:
            c = 1;
            break;
        case 1:
            c = 2;
            break;
        default:
            c = 3;
            break;
        }

    // indent the bodies of if statements
    if (something)
        {
        c = 5;
        b = 10;
        }
     else if (something_else)
        {
        c = 10;
        b = 5;
        }
     else
        {
        c = 20;
        b = 6;
        }

    // omitting the braces is fine if there is only one statement in a body (for loops, if, etc.)
    for (int i = 0; i < 10; i++)
        c = c + 1;

    return c;
    // the nice thing about this style is that every brace lines up perfectly with it's mate
    }

```

* Documentation comments and items broken over multiple lines should be *aligned* with spaces

```
#!c++
class SomeClass
    {
    private:
        int m_some_member;        //!< Documentation for some_member
        int m_some_other_member;  //!< Documentation for some_other_member
    };

template<class BlahBlah> void some_long_func(BlahBlah with_a_really_long_argument_list,
                                             int b,
                                             int c);
```

## Formatting Long Lines

* All code lines should be hand-wrapped so that they are no more than *120 characters* long
* Simply break any excessively long line of code at any natural breaking point to continue on the next line


```
#!c++
cout << "This is a really long message, with "
     << message.length()
     << "Characters in it:"
     << message << endl;
```

* Try to maintain some element of beautiful symmetry in the way the line is broken. For example, the _above_ long message is preferred over the below:

```
#!c++
cout << "This is a really long message, with " << message.length() << "Characters in it:"
   << message << endl;
```

* There are *special rules* for function definitions and/or calls
* If the function definition (or call) cleanly fits within the 120 character limit, leave it all on one line
```
#!c++
int some_function(int arg1, int arg2)
```
* (option 1) If the function definition (or call) goes over the limit, you may be able to fix it by simply putting the template definition on the previous line:
```
#!c++
// go from
template<class Foo, class Bar> int some_really_long_function_name(int with_really_long, Foo argument, Bar lists)
// to
template<class Foo, class Bar>
int some_really_long_function_name(int with_really_long, Foo argument, Bar lists)
```
* (option 2) If the function doesn't have a template specifier, or splitting at that point isn't enough, split out each argument onto a separate line and align them.
```
#!c++
// go from
int some_really_long_function_name(int with_really_long_arguments, int or, int maybe, float there, char are, int just, float a, int lot, char of, int them)
// to
int some_really_long_function_name(int with_really_long_arguments,
                                   int or,
                                   int maybe,
                                   float there,
                                   char are,
                                   int just,
                                   float a,
                                   int lot,
                                   char of,
                                   int them)
```

## Documentation Comments

* Every class, member variable, function, function parameter, macro, etc. *MUST* be documented with *doxygen* comments.
* See http://www.stack.nl/~dimitri/doxygen/docblocks.html
* If you copy an existing file as a template, *DO NOT* simply leave the existing documentation comments there. They apply to the original file, not your new one!
* The best advice that can be given is to write the documentation comments *FIRST* and the _actual code_ *second*. This allows one to formulate their thoughts and write out in English what the code is going to be doing. After thinking through that, writing the actual code is often _much easier_, plus the documentation left for future developers to read is top-notch.

