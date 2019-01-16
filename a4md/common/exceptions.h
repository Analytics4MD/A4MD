#ifndef __EXCEPTIONS_H__
#define __EXCEPTIONS_H__
#include <stdexcept>

class NotImplementedException : public std::logic_error
{
private:

    std::string _text;

    NotImplementedException(const char* message, const char* function)
        :
        std::logic_error("Not Implemented")
    {
        _text = message;
        _text += " : ";
        _text += function;
    };

public:

    NotImplementedException()
        :
        NotImplementedException("Not Implememented", __FUNCTION__)
    {
    }

    NotImplementedException(const char* message)
        :
        NotImplementedException(message, __FUNCTION__)
    {
    }

    virtual const char *what() const noexcept override 
    {
        return _text.c_str();
    }
};

class PythonModuleException : public std::logic_error
{
private:

    std::string _text;

    PythonModuleException(const char* message, const char* function)
        :
        std::logic_error("Python module did not load successfully")
    {
        _text = message;
        _text += " : ";
        _text += function;
    };

public:

    PythonModuleException()
        :
        PythonModuleException("Python module did not load", __FUNCTION__)
    {
    }

    PythonModuleException(const char* message)
        :
        PythonModuleException(message, __FUNCTION__)
    {
    }

    virtual const char *what() const noexcept override 
    {
        return _text.c_str();
    }
};

#endif
